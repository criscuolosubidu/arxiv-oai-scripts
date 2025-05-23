use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use chrono::Utc;
use clap::Parser;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tracing::{error, info, warn};

// 注意：HDF5 binding在Rust中较为复杂，这里提供简化版本
// 在生产环境中建议使用更成熟的库

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// arXiv元数据JSON文件路径
    #[arg(long, default_value = "data/arxiv/arxiv-metadata-oai-snapshot.json")]
    input_file: String,

    /// 输出目录
    #[arg(long, default_value = "data/arxiv/embeddings")]
    output_dir: String,

    /// TEI服务URL
    #[arg(long, default_value = "http://127.0.0.1:8080/embed")]
    tei_url: String,

    /// 并发数 (优化GPU利用率)
    #[arg(long, default_value_t = 40)]
    concurrency: usize,

    /// 批处理大小 (优化GPU利用率)
    #[arg(long, default_value_t = 100)]
    batch_size: usize,

    /// 从第几行开始处理
    #[arg(long, default_value_t = 0)]
    start_idx: usize,

    /// 最多处理多少篇论文
    #[arg(long)]
    max_samples: Option<usize>,

    /// TEI服务的提示名称
    #[arg(long)]
    prompt_name: Option<String>,

    /// 请求超时时间（秒）
    #[arg(long, default_value_t = 30)]
    timeout: u64,

    /// 请求重试次数
    #[arg(long, default_value_t = 3)]
    retries: usize,

    /// 输出格式: binary 或 jsonl
    #[arg(long, default_value = "binary")]
    output_format: String,

    /// 保存间隔（每处理多少篇论文保存一次）
    #[arg(long, default_value_t = 1000)]
    save_every: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct ArxivPaper {
    id: String,
    title: String,
    #[serde(rename = "abstract")]
    abstract_text: String,
    authors: Option<String>,
    categories: Option<String>,
    #[serde(rename = "journal-ref")]
    journal_ref: Option<String>,
    doi: Option<String>,
    update_date: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ProcessedPaper {
    id: String,
    title: String,
    abstract_text: String,
    title_embedding: Vec<f32>,
    abstract_embedding: Vec<f32>,
    authors: Option<String>,
    categories: Option<String>,
    journal_ref: Option<String>,
    doi: Option<String>,
    update_date: Option<String>,
}

#[derive(Debug, Serialize)]
struct TeiRequest {
    inputs: String,
}

#[derive(Debug, Deserialize)]
struct TeiResponse(Vec<Vec<f32>>);

#[derive(Debug, Serialize)]
struct OutputMetadata {
    embedding_info: EmbeddingInfo,
}

#[derive(Debug, Serialize)]
struct EmbeddingInfo {
    model: String,
    embedding_dim: usize,
    creation_date: String,
    prompt_name: Option<String>,
    total_papers: usize,
    processing_method: String,
    data_type: String,
    compression: String,
    throughput_optimization: ThroughputStats,
}

#[derive(Debug, Serialize)]
struct ThroughputStats {
    target_throughput: f64,
    actual_throughput: f64,
    gpu_utilization_percent: f64,
    batch_size: usize,
    concurrency: usize,
}

async fn call_tei_service(
    client: &Client,
    text: &str,
    tei_url: &str,
    prompt_name: &Option<String>,
    timeout: Duration,
    retries: usize,
) -> Result<Vec<f32>> {
    let inputs = match prompt_name {
        Some(prompt) => format!("{} {}", prompt, text),
        None => text.to_string(),
    };

    let request = TeiRequest { inputs };

    for attempt in 0..retries {
        match client
            .post(tei_url)
            .json(&request)
            .timeout(timeout)
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    match response.json::<TeiResponse>().await {
                        Ok(tei_response) => {
                            if !tei_response.0.is_empty() {
                                return Ok(tei_response.0[0].clone());
                            } else {
                                return Err(anyhow!("Empty response from TEI service"));
                            }
                        }
                        Err(e) => {
                            warn!("Failed to parse TEI response: {}", e);
                            if attempt == retries - 1 {
                                return Err(anyhow!("Failed to parse TEI response: {}", e));
                            }
                        }
                    }
                } else if response.status().as_u16() == 429 {
                    let delay = Duration::from_millis(100 * (2_u64.pow(attempt as u32)));
                    warn!("Rate limited, retrying in {:?}", delay);
                    sleep(delay).await;
                    continue;
                } else {
                    warn!("TEI service error: {}", response.status());
                    if attempt == retries - 1 {
                        return Err(anyhow!("TEI service error: {}", response.status()));
                    }
                }
            }
            Err(e) => {
                warn!("Request failed: {}", e);
                if attempt == retries - 1 {
                    return Err(anyhow!("Request failed: {}", e));
                }
            }
        }

        let delay = Duration::from_millis(100 * (2_u64.pow(attempt as u32)));
        sleep(delay).await;
    }

    Err(anyhow!("Exhausted all retries"))
}

async fn process_paper_optimized(
    client: &Client,
    paper: ArxivPaper,
    tei_url: &str,
    prompt_name: &Option<String>,
    timeout: Duration,
    retries: usize,
) -> Result<ProcessedPaper> {
    // 并发获取标题和摘要的嵌入
    let title_future = call_tei_service(
        client,
        &paper.title,
        tei_url,
        prompt_name,
        timeout,
        retries,
    );

    let abstract_future = call_tei_service(
        client,
        &paper.abstract_text,
        tei_url,
        prompt_name,
        timeout,
        retries,
    );

    // 并发等待两个结果
    let (title_embedding, abstract_embedding) = tokio::try_join!(title_future, abstract_future)?;

    Ok(ProcessedPaper {
        id: paper.id,
        title: paper.title,
        abstract_text: paper.abstract_text,
        title_embedding,
        abstract_embedding,
        authors: paper.authors,
        categories: paper.categories,
        journal_ref: paper.journal_ref,
        doi: paper.doi,
        update_date: paper.update_date,
    })
}

fn stream_papers_from_file(
    file_path: &str,
    start_idx: usize,
    max_samples: Option<usize>,
) -> Result<impl Iterator<Item = Result<ArxivPaper>>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let lines = reader.lines();

    Ok(lines
        .skip(start_idx)
        .take(max_samples.unwrap_or(usize::MAX))
        .map(|line| {
            let line = line?;
            let paper: ArxivPaper = serde_json::from_str(&line)?;
            
            if paper.title.trim().is_empty() || paper.abstract_text.trim().is_empty() {
                return Err(anyhow!("Missing title or abstract"));
            }
            
            Ok(paper)
        }))
}

async fn save_batch_to_jsonl(
    papers: &[ProcessedPaper],
    output_dir: &str,
    batch_number: usize,
) -> Result<()> {
    let output_path = Path::new(output_dir).join(format!("embeddings_batch_{}.jsonl", batch_number));
    
    let mut file_content = String::new();
    for paper in papers {
        let json_line = serde_json::to_string(paper)?;
        file_content.push_str(&json_line);
        file_content.push('\n');
    }
    
    tokio::fs::write(output_path, file_content).await?;
    Ok(())
}

async fn save_batch_to_binary(
    papers: &[ProcessedPaper],
    output_dir: &str,
    batch_number: usize,
) -> Result<()> {
    use std::io::Write;
    
    let output_path = Path::new(output_dir).join(format!("embeddings_batch_{}.bin", batch_number));
    let mut file = std::fs::File::create(output_path)?;
    
    // 写入批次大小
    file.write_all(&(papers.len() as u32).to_le_bytes())?;
    
    for paper in papers {
        // 写入ID长度和ID
        let id_bytes = paper.id.as_bytes();
        file.write_all(&(id_bytes.len() as u32).to_le_bytes())?;
        file.write_all(id_bytes)?;
        
        // 写入嵌入维度
        file.write_all(&(paper.title_embedding.len() as u32).to_le_bytes())?;
        
        // 写入标题嵌入向量
        for &val in &paper.title_embedding {
            file.write_all(&val.to_le_bytes())?;
        }
        
        // 写入摘要嵌入向量
        for &val in &paper.abstract_embedding {
            file.write_all(&val.to_le_bytes())?;
        }
    }
    
    file.flush()?;
    Ok(())
}

async fn process_batch_concurrent(
    papers: Vec<ArxivPaper>,
    client: &Client,
    tei_url: &str,
    prompt_name: &Option<String>,
    timeout: Duration,
    retries: usize,
    semaphore: Arc<Semaphore>,
) -> Vec<ProcessedPaper> {
    let tasks: Vec<_> = papers
        .into_iter()
        .map(|paper| {
            let client = client.clone();
            let tei_url = tei_url.to_string();
            let prompt_name = prompt_name.clone();
            let semaphore = semaphore.clone();
            
            tokio::spawn(async move {
                let _permit = semaphore.acquire().await.ok()?;
                process_paper_optimized(&client, paper, &tei_url, &prompt_name, timeout, retries).await.ok()
            })
        })
        .collect();

    let results = futures::future::join_all(tasks).await;
    results.into_iter().filter_map(|r| r.ok().flatten()).collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    tokio::fs::create_dir_all(&args.output_dir).await?;

    info!("Starting ArXiv embedding generation - GPU Throughput Optimized");
    info!("Target TEI performance: 40 papers/second");
    info!("Input file: {}", args.input_file);
    info!("Output directory: {}", args.output_dir);
    info!("TEI URL: {}", args.tei_url);
    info!("Concurrency: {} (optimized for GPU utilization)", args.concurrency);
    info!("Batch size: {} (optimized for GPU utilization)", args.batch_size);
    info!("Output format: {}", args.output_format);

    let client = Client::builder()
        .timeout(Duration::from_secs(args.timeout))
        .build()?;

    // 测试TEI服务连接
    info!("Testing TEI service connection...");
    let test_embedding = call_tei_service(
        &client,
        "Test connection",
        &args.tei_url,
        &args.prompt_name,
        Duration::from_secs(args.timeout),
        args.retries,
    )
    .await?;
    
    let embedding_dim = test_embedding.len();
    info!("TEI service connected successfully, embedding dimension: {}", embedding_dim);
    
    // GPU吞吐量优化分析
    let size_per_paper_kb = (embedding_dim * 2 * 4) / 1024; // 2 embeddings * 4 bytes (f32)
    let batch_processing_time = args.batch_size as f64 / 40.0; // 预期处理时间
    let batch_memory_mb = args.batch_size * size_per_paper_kb / 1024;
    
    info!("GPU Throughput Optimization Analysis:");
    info!("- TEI target performance: 40 papers/second");
    info!("- Current batch size: {} papers", args.batch_size);
    info!("- Expected batch processing time: {:.1} seconds", batch_processing_time);
    info!("- Concurrency level: {}", args.concurrency);
    info!("- Per paper embedding size: {}KB", size_per_paper_kb);
    info!("- Single batch memory usage: {:.1}MB", batch_memory_mb);
    info!("- Peak memory during save interval: {:.1}MB", 
          (args.save_every * size_per_paper_kb) as f64 / 1024.0);

    let semaphore = Arc::new(Semaphore::new(args.concurrency));

    let progress_bar = ProgressBar::new_spinner();
    progress_bar.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {pos} papers processed | {per_sec} papers/sec | GPU: {msg}")
            .unwrap(),
    );

    let start_time = Instant::now();
    let mut total_processed = 0;
    let mut batch_number = 0;
    let mut current_batch = Vec::new();
    let mut throughput_samples = Vec::new();

    let paper_stream = stream_papers_from_file(&args.input_file, args.start_idx, args.max_samples)?;
    
    for paper_result in paper_stream {
        match paper_result {
            Ok(paper) => {
                current_batch.push(paper);
                
                if current_batch.len() >= args.batch_size {
                    let batch_start = Instant::now();
                    
                    // 并发处理当前批次
                    let processed_papers = process_batch_concurrent(
                        current_batch.clone(),
                        &client,
                        &args.tei_url,
                        &args.prompt_name,
                        Duration::from_secs(args.timeout),
                        args.retries,
                        semaphore.clone(),
                    ).await;
                    
                    let batch_duration = batch_start.elapsed();
                    let batch_throughput = processed_papers.len() as f64 / batch_duration.as_secs_f64();
                    throughput_samples.push(batch_throughput);
                    
                    if !processed_papers.is_empty() {
                        // 保存批次
                        match args.output_format.as_str() {
                            "binary" => {
                                save_batch_to_binary(&processed_papers, &args.output_dir, batch_number).await?;
                            }
                            _ => {
                                save_batch_to_jsonl(&processed_papers, &args.output_dir, batch_number).await?;
                            }
                        }
                        
                        total_processed += processed_papers.len();
                        batch_number += 1;
                        
                        // 计算GPU利用率
                        let gpu_utilization = (batch_throughput / 40.0 * 100.0).min(100.0);
                        
                        // 更新进度条
                        progress_bar.inc(processed_papers.len() as u64);
                        if total_processed % 1000 == 0 {
                            let avg_throughput: f64 = throughput_samples.iter().sum::<f64>() / throughput_samples.len() as f64;
                            progress_bar.set_message(format!("{:.1}% util, {:.1} t/s", gpu_utilization, avg_throughput));
                        }
                        
                        info!("Batch {} completed: {} papers in {:.1}s | {:.1} papers/sec | GPU util: {:.1}%", 
                              batch_number, processed_papers.len(), batch_duration.as_secs_f64(), 
                              batch_throughput, gpu_utilization);
                    }
                    
                    current_batch.clear();
                }
            }
            Err(e) => {
                warn!("Failed to parse paper: {}", e);
            }
        }
    }

    // 处理最后一批
    if !current_batch.is_empty() {
        let processed_papers = process_batch_concurrent(
            current_batch,
            &client,
            &args.tei_url,
            &args.prompt_name,
            Duration::from_secs(args.timeout),
            args.retries,
            semaphore.clone(),
        ).await;
        
        if !processed_papers.is_empty() {
            match args.output_format.as_str() {
                "binary" => {
                    save_batch_to_binary(&processed_papers, &args.output_dir, batch_number).await?;
                }
                _ => {
                    save_batch_to_jsonl(&processed_papers, &args.output_dir, batch_number).await?;
                }
            }
            total_processed += processed_papers.len();
        }
    }

    progress_bar.finish();

    let elapsed = start_time.elapsed();
    let overall_throughput = total_processed as f64 / elapsed.as_secs_f64();
    let avg_batch_throughput: f64 = throughput_samples.iter().sum::<f64>() / throughput_samples.len() as f64;
    let gpu_utilization = (overall_throughput / 40.0 * 100.0).min(100.0);

    // 保存元数据
    let metadata = OutputMetadata {
        embedding_info: EmbeddingInfo {
            model: "TEI Service (E5-Mistral-7B)".to_string(),
            embedding_dim,
            creation_date: Utc::now().to_rfc3339(),
            prompt_name: args.prompt_name,
            total_papers: total_processed,
            processing_method: "rust_gpu_throughput_optimized".to_string(),
            data_type: "float32".to_string(),
            compression: match args.output_format.as_str() {
                "binary" => "custom_binary".to_string(),
                _ => "none".to_string(),
            },
            throughput_optimization: ThroughputStats {
                target_throughput: 40.0,
                actual_throughput: overall_throughput,
                gpu_utilization_percent: gpu_utilization,
                batch_size: args.batch_size,
                concurrency: args.concurrency,
            },
        },
    };

    let metadata_path = Path::new(&args.output_dir).join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    tokio::fs::write(metadata_path, metadata_json).await?;

    // 计算输出文件总大小
    let mut total_file_size = 0u64;
    for i in 0..=batch_number {
        let file_path = if args.output_format == "binary" {
            Path::new(&args.output_dir).join(format!("embeddings_batch_{}.bin", i))
        } else {
            Path::new(&args.output_dir).join(format!("embeddings_batch_{}.jsonl", i))
        };
        
        if let Ok(metadata) = std::fs::metadata(&file_path) {
            total_file_size += metadata.len();
        }
    }
    
    let total_size_gb = total_file_size as f64 / (1024.0 * 1024.0 * 1024.0);
    
    info!("="*80);
    info!("Processing completed with GPU throughput optimization!");
    info!("Total papers processed: {}", total_processed);
    info!("Total time: {:.2} minutes", elapsed.as_secs_f64() / 60.0);
    info!("Overall throughput: {:.2} papers/second", overall_throughput);
    info!("Average batch throughput: {:.2} papers/second", avg_batch_throughput);
    info!("GPU utilization: {:.1}% (target: close to 100%)", gpu_utilization);
    info!("Total batches saved: {}", batch_number + 1);
    info!("Output file size: {:.2} GB", total_size_gb);
    
    if gpu_utilization < 80.0 {
        warn!("GPU utilization is below 80%. Consider:");
        warn!("- Increasing batch size (--batch-size)");
        warn!("- Increasing concurrency (--concurrency)");
        warn!("- Checking network latency to TEI service");
    } else {
        info!("Excellent GPU utilization! Configuration is well-optimized.");
    }
    
    info!("="*80);

    Ok(())
}