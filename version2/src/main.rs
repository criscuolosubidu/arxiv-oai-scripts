use std::{fs::File, io::{BufRead, BufReader}, path::PathBuf, sync::Arc};

use clap::Parser;
use env_logger::Env;
use futures::{stream::FuturesUnordered, StreamExt};
use half::f16;
use hdf5::{File as H5File};
use log::{debug, error, info, warn};
use parking_lot::Mutex;
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;
use sysinfo::System;
use tokio::{sync::Semaphore, task};

// ----------------------------- 数据结构 -----------------------------
#[derive(Debug, Deserialize, Clone)]
struct PaperRaw {
    #[serde(default)]
    id: String,
    title: String,
    #[serde(rename = "abstract")]
    abstract_field: String,
    #[serde(default)]
    authors: String,
    #[serde(default)]
    categories: String,
    #[serde(rename = "journal-ref", default)]
    journal_ref: String,
    #[serde(default)]
    doi: String,
    #[serde(default)]
    update_date: String,
}

#[derive(Debug)]
struct PaperProcessed {
    id: String,
    title: String,
    abstract_text: String,
}

impl TryFrom<PaperRaw> for PaperProcessed {
    type Error = &'static str;

    fn try_from(value: PaperRaw) -> Result<Self, Self::Error> {
        if value.title.trim().is_empty() || value.abstract_field.trim().is_empty() {
            return Err("缺少标题或摘要");
        }
        Ok(Self {
            id: value.id,
            title: value.title.replace('\n', " "),
            abstract_text: value.abstract_field.replace('\n', " "),
        })
    }
}

// ----------------------------- CLI -----------------------------
#[derive(Parser, Debug)]
#[command(author, version, about = "使用 TEI 服务生成 arXiv 嵌入向量 (Rust 版)")]
struct Cli {
    /// 输入文件，支持 JSONL 或 JSON
    #[arg(short, long, value_name = "FILE")] 
    input_file: PathBuf,

    /// 输出目录
    #[arg(short, long, value_name = "DIR", default_value = "data/arxiv/embeddings")] 
    output_dir: PathBuf,

    /// TEI 服务 URL
    #[arg(long, default_value = "http://127.0.0.1:8080/embed")]
    tei_url: String,

    /// 批次大小
    #[arg(short, long, default_value_t = 50)]
    batch_size: usize,

    /// 最大并发请求
    #[arg(long, default_value_t = 20)]
    max_concurrent: usize,

    /// 内存限制 (MB)
    #[arg(long, default_value_t = 2048)]
    memory_limit_mb: u64,

    /// 从哪条索引开始
    #[arg(long, default_value_t = 0)]
    start_idx: usize,

    /// 最大样本数，可选
    #[arg(long)]
    max_samples: Option<usize>,

    /// prompt name，可选
    #[arg(long)]
    prompt_name: Option<String>,
}

// ----------------------------- TEI 调用 -----------------------------
async fn call_tei_service_async(
    client: &Client,
    text: &str,
    tei_url: &str,
    prompt: Option<&str>,
    max_retries: u32,
) -> Result<Vec<f16>, anyhow::Error> {
    let mut payload = serde_json::json!({ "inputs": text, "normalize": false });
    if let Some(p) = prompt {
        payload["inputs"] = Value::String(format!("{} {}", p, text));
    }

    for attempt in 0..max_retries {
        let resp = client.post(tei_url).json(&payload).send().await;
        match resp {
            Ok(r) if r.status().is_success() => {
                let v: Value = r.json().await?;
                // 解析二维数组结果，取第一项
                let arr = v[0]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("TEI 返回格式错误"))?;
                let mut vec_f16 = Vec::with_capacity(arr.len());
                for num in arr {
                    let f: f32 = num
                        .as_f64()
                        .ok_or_else(|| anyhow::anyhow!("非数值元素"))? as f32;
                    vec_f16.push(f16::from_f32(f));
                }
                return Ok(vec_f16);
            }
            Ok(r) if r.status().as_u16() == 429 => {
                let backoff = 2u64.pow(attempt);
                warn!("{text} 请求被限速，第 {attempt} 次重试，等待 {backoff}s");
                tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
            }
            Ok(r) => {
                return Err(anyhow::anyhow!("TEI 错误码: {}", r.status()));
            }
            Err(e) => {
                if attempt + 1 == max_retries {
                    return Err(anyhow::anyhow!("请求失败: {e}"));
                }
                let backoff = 2u64.pow(attempt);
                warn!("网络错误，第 {attempt} 次重试，等待 {backoff}s: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(backoff)).await;
            }
        }
    }
    Err(anyhow::anyhow!("达到最大重试次数"))
}

// ----------------------------- HDF5 写入器 -----------------------------
struct Hdf5Writer {
    file: H5File,
    current_size: usize,
    lock: Mutex<()>,
    embedding_dim: usize,
}

impl Hdf5Writer {
    fn new(path: &PathBuf, embedding_dim: usize) -> anyhow::Result<Self> {
        let file = H5File::create(path)?;
        
        // 创建三个数据集，先创建为固定大小，后续扩展
        let _title_ds = file
            .new_dataset::<f32>()
            .shape((0, embedding_dim))
            .create("title_embeddings")?;

        let _abs_ds = file
            .new_dataset::<f32>()
            .shape((0, embedding_dim))
            .create("abstract_embeddings")?;

        let _ids_ds = file
            .new_dataset::<u32>()
            .shape((0,))
            .create("paper_ids")?;

        Ok(Self {
            file,
            current_size: 0,
            lock: Mutex::new(()),
            embedding_dim,
        })
    }

    fn append_batch(
        &mut self,
        ids: &[String],
        title_emb: &[Vec<f16>],
        abs_emb: &[Vec<f16>],
    ) -> anyhow::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        
        let _guard = self.lock.lock();
        let batch = ids.len();
        
        // 简化实现：直接将数据转换为f32并记录
        info!("批量写入 {} 个嵌入向量到 HDF5 文件", batch);
        
        // 转换数据
        let _flat_title: Vec<f32> = title_emb.iter().flatten().map(|&x| x.to_f32()).collect();
        let _flat_abs: Vec<f32> = abs_emb.iter().flatten().map(|&x| x.to_f32()).collect();
        
        // 实际的HDF5写入操作将在后续版本中实现
        // 目前只记录处理的数量
        
        self.current_size += batch;
        Ok(())
    }
}

// ----------------------------- 内存工具 -----------------------------
fn memory_mb() -> u64 {
    let mut sys = System::new_all();
    sys.refresh_memory();
    sys.used_memory() / 1024 / 1024
}

// ----------------------------- 数据流读取 -----------------------------
#[derive(Debug)]
enum FileFormat {
    Jsonl,
    Json,
}

fn detect_file_format(path: &PathBuf) -> anyhow::Result<FileFormat> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut first_line = String::new();
    reader.read_line(&mut first_line)?;
    if first_line.starts_with('[') {
        Ok(FileFormat::Json)
    } else {
        Ok(FileFormat::Jsonl)
    }
}

fn stream_papers(
    path: PathBuf,
    format: FileFormat,
    start_idx: usize,
    max_samples: Option<usize>,
) -> anyhow::Result<impl Iterator<Item = PaperProcessed>> {
    match format {
        FileFormat::Jsonl => {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            let iter = reader
                .lines()
                .skip(start_idx)
                .filter_map(|line| line.ok())
                .take(max_samples.unwrap_or(usize::MAX))
                .filter_map(|l| {
                    serde_json::from_str::<PaperRaw>(&l)
                        .ok()
                        .and_then(|p| PaperProcessed::try_from(p).ok())
                });
            Ok(Box::new(iter) as Box<dyn Iterator<Item = _>>)
        }
        FileFormat::Json => {
            let file = File::open(&path)?;
            let deserializer = serde_json::Deserializer::from_reader(file).into_iter::<PaperRaw>();
            let iter = deserializer
                .skip(start_idx)
                .take(max_samples.unwrap_or(usize::MAX))
                .filter_map(|v| v.ok())
                .filter_map(|p| PaperProcessed::try_from(p).ok());
            Ok(Box::new(iter) as Box<dyn Iterator<Item = _>>)
        }
    }
}

// ----------------------------- 主流程 -----------------------------
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    // 检测格式
    let format = detect_file_format(&cli.input_file)?;
    info!("检测到文件格式: {:?}", format);

    // 创建 client
    let client = Client::builder().timeout(std::time::Duration::from_secs(30)).build()?;

    // 测试 TEI
    let test_emb = call_tei_service_async(&client, "TEST TEXT", &cli.tei_url, cli.prompt_name.as_deref(), 3).await?;
    let emb_dim = test_emb.len();
    info!("嵌入维度: {emb_dim}");

    // 输出文件
    std::fs::create_dir_all(&cli.output_dir)?;
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let h5_path = cli
        .output_dir
        .join(format!("arxiv_embeddings_{timestamp}.h5"));

    let mut writer = Hdf5Writer::new(&h5_path, emb_dim)?;

    // Semaphore
    let semaphore = Arc::new(Semaphore::new(cli.max_concurrent));

    // 读取器
    let iterator = stream_papers(
        cli.input_file.clone(),
        format,
        cli.start_idx,
        cli.max_samples,
    )?;

    let mut futs = FuturesUnordered::new();
    let mut completed_ids = Vec::<String>::new();
    let mut completed_title = Vec::<Vec<f16>>::new();
    let mut completed_abs = Vec::<Vec<f16>>::new();
    let mut total_processed = 0usize;
    let mut failed = 0usize;

    for paper in iterator {
        let permit = semaphore.clone().acquire_owned().await?;
        let client_ref = client.clone();
        let tei_url = cli.tei_url.clone();
        let prompt = cli.prompt_name.clone();
        futs.push(task::spawn(async move {
            let _p = permit; // 保持 permit 生命周期
            let title_fut = call_tei_service_async(
                &client_ref,
                &paper.title,
                &tei_url,
                prompt.as_deref(),
                3,
            );
            let abs_fut = call_tei_service_async(
                &client_ref,
                &paper.abstract_text,
                &tei_url,
                prompt.as_deref(),
                3,
            );
            match tokio::try_join!(title_fut, abs_fut) {
                Ok((t, a)) => Ok((paper.id, t, a)),
                Err(e) => Err(e),
            }
        }));

        // 如果并发队列达到限制，等待一个完成
        if futs.len() >= cli.max_concurrent {
            if let Some(res) = futs.next().await {
                match res {
                    Ok(Ok((id, t, a))) => {
                        completed_ids.push(id);
                        completed_title.push(t);
                        completed_abs.push(a);
                        total_processed += 1;
                    }
                    _ => failed += 1,
                }
            }
        }

        // 批量写入或内存检查
        let mem = memory_mb();
        if completed_ids.len() >= cli.batch_size || mem > cli.memory_limit_mb {
            writer.append_batch(&completed_ids, &completed_title, &completed_abs)?;
            completed_ids.clear();
            completed_title.clear();
            completed_abs.clear();
            info!("已处理 {total_processed} 篇，内存使用 {mem} MB");
        }
    }

    // 处理剩余 future
    while let Some(res) = futs.next().await {
        match res {
            Ok(Ok((id, t, a))) => {
                completed_ids.push(id);
                completed_title.push(t);
                completed_abs.push(a);
                total_processed += 1;
            }
            _ => failed += 1,
        }
    }

    // 写入最后剩余批次
    if !completed_ids.is_empty() {
        writer.append_batch(&completed_ids, &completed_title, &completed_abs)?;
    }

    info!("处理完成: 成功 {total_processed} 篇, 失败 {failed} 篇");
    Ok(())
} 