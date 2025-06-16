import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import gc


def compare_tei_vs_sentence_transformers():
    """比较TEI服务、sentence-transformers和transformers的向量输出"""

    model_id = "/home/ubuntu/projects/dev/arxiv-oai-scripts/models/e5-mistral-7b-instruct"
    
    # 配置
    tei_url = "http://127.0.0.1:8080/embed"
    test_texts = [
        "Attention is all you need",
        # "Large language models have revolutionized natural language processing"
    ]
    
    print("="*80)
    print("比较TEI服务、sentence-transformers和transformers的向量输出")
    print("="*80)
    
    # 检查GPU内存
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name()}")
        print(f"初始GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
    
    # 定义TEI服务调用函数
    def call_tei_service(text):
        payload = {"inputs": text, "normalize": False}
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(tei_url, json=payload, headers=headers)
            if response.status_code == 200:
                # 明确指定数据类型为float32，与其他方法保持一致
                return np.array(response.json()[0], dtype=np.float16)
            else:
                print(f"TEI服务错误，状态码: {response.status_code}")
                return None
        except Exception as e:
            print(f"调用TEI服务失败: {e}")
            return None
    
    # 存储所有结果
    all_results = []
    
    # 对每个测试文本进行处理
    for i, text in enumerate(test_texts):
        print(f"\n测试文本 {i+1}: {text}")
        print("-" * 80)
        
        result = {"text": text, "embeddings": {}}
        
        # 1. 获取TEI向量
        print("正在获取TEI向量...")
        tei_embedding = call_tei_service(text)
        if tei_embedding is None:
            print("跳过此文本，TEI服务调用失败")
            continue
        result["embeddings"]["tei"] = tei_embedding
        
        # 2. 获取sentence-transformers向量
        print("正在加载sentence-transformers模型...")
        try:
            st_model = SentenceTransformer(model_id)
            if torch.cuda.is_available():
                print(f"ST模型加载后GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
            
            print("正在计算sentence-transformers向量...")
            st_embedding = st_model.encode([text])[0]
            result["embeddings"]["st"] = st_embedding
            
            # 卸载sentence-transformers模型
            print("正在卸载sentence-transformers模型...")
            del st_model
            torch.cuda.empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                print(f"ST模型卸载后GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
                
        except Exception as e:
            print(f"sentence-transformers处理失败: {e}")
            continue
        
        # 3. 获取transformers向量
        print("正在加载transformers模型...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # 使用FlashAttention2加载模型
            model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.float16,  # 使用半精度节省显存
                attn_implementation="flash_attention_2",  # 使用FlashAttention2
                device_map="cuda:0"  # 自动设备映射
            )
            model.eval()
            
            if torch.cuda.is_available():
                print(f"TF模型加载后GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
            
            print("正在计算transformers向量...")
            # 分词
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4096)
            
            # 将inputs移动到模型所在的设备
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 获取模型输出
            with torch.no_grad():
                outputs = model(**inputs)
                
            # 获取最后一层隐藏状态
            last_hidden_states = outputs.last_hidden_state
            
            # 使用mean pooling获取句子嵌入
            attention_mask = inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sentence_embedding = sum_embeddings / sum_mask
            
            tf_embedding = sentence_embedding.cpu().numpy()[0]
            result["embeddings"]["tf"] = tf_embedding
            
            # 卸载transformers模型
            print("正在卸载transformers模型...")
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            if torch.cuda.is_available():
                print(f"TF模型卸载后GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
                
        except Exception as e:
            print(f"transformers处理失败: {e}")
            continue
        
        all_results.append(result)
    
    # 分析所有结果
    print("\n" + "="*80)
    print("分析结果")
    print("="*80)
    
    for i, result in enumerate(all_results):
        text = result["text"]
        embeddings = result["embeddings"]
        
        print(f"\n测试文本 {i+1}: {text}")
        print("-" * 80)
        
        # 检查是否有所有三种嵌入
        if len(embeddings) < 3:
            print("缺少某些嵌入向量，跳过分析")
            continue
        
        tei_embedding = embeddings["tei"]
        st_embedding = embeddings["st"]
        tf_embedding = embeddings["tf"]
        
        # 基本信息
        print(f"TEI向量维度: {tei_embedding.shape}, 数据类型: {tei_embedding.dtype}")
        print(f"ST向量维度: {st_embedding.shape}, 数据类型: {st_embedding.dtype}")
        print(f"TF向量维度: {tf_embedding.shape}, 数据类型: {tf_embedding.dtype}")
        
        # 计算范数
        tei_norm = np.linalg.norm(tei_embedding)
        st_norm = np.linalg.norm(st_embedding)
        tf_norm = np.linalg.norm(tf_embedding)
        
        print(f"TEI向量范数: {tei_norm:.6f}")
        print(f"ST向量范数: {st_norm:.6f}")
        print(f"TF向量范数: {tf_norm:.6f}")
        
        # 计算余弦相似度
        tei_st_sim = np.dot(tei_embedding, st_embedding) / (tei_norm * st_norm)
        tei_tf_sim = np.dot(tei_embedding, tf_embedding) / (tei_norm * tf_norm)
        st_tf_sim = np.dot(st_embedding, tf_embedding) / (st_norm * tf_norm)
        
        print(f"TEI vs ST 余弦相似度: {tei_st_sim:.6f}")
        print(f"TEI vs TF 余弦相似度: {tei_tf_sim:.6f}")
        print(f"ST vs TF 余弦相似度: {st_tf_sim:.6f}")
        
        # 显示前50个维度的值
        print(f"\n前50个维度的值比较:")
        print("维度\tTEI值\t\tST值\t\tTF值\t\t|TEI-ST|\t|TEI-TF|\t|ST-TF|")
        print("-" * 100)
        
        for j in range(min(50, len(tei_embedding), len(st_embedding), len(tf_embedding))):
            tei_st_diff = abs(tei_embedding[j] - st_embedding[j])
            tei_tf_diff = abs(tei_embedding[j] - tf_embedding[j])
            st_tf_diff = abs(st_embedding[j] - tf_embedding[j])
            print(f"{j:3d}\t{tei_embedding[j]:8.6f}\t{st_embedding[j]:8.6f}\t{tf_embedding[j]:8.6f}\t{tei_st_diff:8.6f}\t{tei_tf_diff:8.6f}\t{st_tf_diff:8.6f}")
        
        # 统计差异
        tei_st_diff = np.abs(tei_embedding - st_embedding)
        tei_tf_diff = np.abs(tei_embedding - tf_embedding)
        st_tf_diff = np.abs(st_embedding - tf_embedding)
        
        print(f"\n差异统计:")
        print(f"TEI vs ST - 平均: {np.mean(tei_st_diff):.8f}, 最大: {np.max(tei_st_diff):.8f}, 标准差: {np.std(tei_st_diff):.8f}")
        print(f"TEI vs TF - 平均: {np.mean(tei_tf_diff):.8f}, 最大: {np.max(tei_tf_diff):.8f}, 标准差: {np.std(tei_tf_diff):.8f}")
        print(f"ST vs TF - 平均: {np.mean(st_tf_diff):.8f}, 最大: {np.max(st_tf_diff):.8f}, 标准差: {np.std(st_tf_diff):.8f}")
        
        # 相似度评估
        def evaluate_similarity(sim, name):
            if sim > 0.99:
                return f"✓ {name} 高度相似 ({sim:.6f})"
            elif sim > 0.95:
                return f"⚠ {name} 相似但有差异 ({sim:.6f})"
            else:
                return f"✗ {name} 差异较大 ({sim:.6f})"
        
        print(f"\n相似度评估:")
        print(evaluate_similarity(tei_st_sim, "TEI vs ST"))
        print(evaluate_similarity(tei_tf_sim, "TEI vs TF"))
        print(evaluate_similarity(st_tf_sim, "ST vs TF"))


if __name__ == "__main__":
    # analyze()
    # merge_batch_metadata_files()
    compare_tei_vs_sentence_transformers()
    