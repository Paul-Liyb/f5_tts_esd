# 情感语音合成-基于F5-TTS在ESD上微调

## 一、项目概述

### 1.1 任务定义
- **输入**：参考文本 + 参考音频，待生成文本 + 所需情感条件
- **输出**：对应声纹的对应情感语音音频

### 1.2 代表数据集
- ESD (Emotional Speech Database)
- 10种中文音色 + 10种英文音色
- 每人350种句子，每种句子5种情绪

### 1.3 项目结构
```
model/
├── src/f5_tts/
│   ├── configs/
│   │   ├── F5TTS_ESD_CH.yaml    # 中文ESD配置
│   │   └── F5TTS_ESD_EN.yaml    # 英文ESD配置
│   ├── model/
│   │   ├── cfm.py               # Flow Matching核心模型
│   │   ├── dataset.py           # ESD数据集处理
│   │   ├── trainer.py           # 训练器
│   │   └── backbones/
│   │       └── dit.py           # DiT骨干网络
│   ├── train/
│   │   └── train.py             # 训练入口
│   └── infer/
│       └── infer_cli.py         # 推理接口
├── data/
│   └── ED_EN_CUSTOM/            # 预处理数据
```

## 二、F5-TTS基础

### 2.1 核心范式：Flow Matching
- 高效的非自回归生成模型
- 替代传统扩散模型去噪过程
- 训练和推理速度更快
- 结合Transformer序列建模能力

**核心代码实现** (`cfm.py`):
```python
# Flow Matching核心公式
t = time.unsqueeze(-1).unsqueeze(-1)
φ = (1 - t) * x0 + t * x1  # 插值
flow = x1 - x0             # 目标流
```

### 2.2 核心能力：零样本语音克隆
- 通过"语音填充(Speech Infilling)"任务
- 仅需一段参考音频+文本即可克隆声音特质
- 复刻音色、韵律甚至隐含情感

### 2.3 局限性
- 能非常好地复刻音色
- 在情感控制方面存在局限

## 三、任务实施过程

### 3.1 数据集转换

#### ESD数据集结构
- 保留情感标签
- 保留说话人标签
- 保留句子标签
- 满足F5-TTS数据要求

**情感映射定义** (`dataset.py`):
```python
self.emo_map = {"Angry": 0, "Happy": 1, "Neutral": 2, "Sad": 3, "Surprise": 4}
```

#### 配置文件示例 (`F5TTS_ESD_CH.yaml`):
```yaml
model:
  arch:
    emotion_num_embeds: 6  # 5种情感 + 1个null情感
```

### 3.2 初步探索与反思

#### 情感于时间注入（效果差）
**可能原因**：
- 情感信息混合在声纹信息中
- F5-TTS零样本泛化能力过强
- 从参考音频中捕捉到情感+声纹信息
- ESD数据集数据量过小，模型难以有效学习

## 四、核心难点

1. **声纹与情感信息难以分离**
2. **情感特征融合的方式**
3. **新的损失函数引导而非简单的重建**
4. **有效的数据学习方法与任务设计**

## 五、解决方法

### 5.1 数据采样策略

**核心思想**：使用同一发声者的不同音频，随机对情感和句子内容进行采样，让模型无法根据第一部分声音信息获取第二部分情感特征，迫使模型利用Emotion标签。

**实现代码** (`dataset.py`):
```python
def __getitem__(self, index):
    # 采样第一段音频
    row = self.data[current_index]
    speaker_id = row["speaker_id"]
    emotion = self.emo_map.get(row["emotion"], 2)
    
    # 采样第二段音频：同一说话人，随机情感，随机句子
    emotion_2 = self.sample_emotion(self.data_mapping, speaker_id)
    phrase_id_2 = self.sample_phrase(self.data_mapping, speaker_id, emotion_2)
    index_2 = self.data_mapping[speaker_id][emotion_2][phrase_id_2][0]
    
    return {
        "mel_spec": torch.cat([mel_spec, mel_spec_2], dim=1),
        "text": text + text_2,
        "emotion": torch.cat([emotion_tensor, emotion_tensor_2], dim=0),
        "mel_len_1": mel_spec.shape[-1],
        "text_len_1": len(text),
    }
```

### 5.2 对比学习

**核心思想**：增加无emotion条件下的生成结果作为负样本，减小正负样本与生成目标之间的loss，提高正负样本对之间的距离，增强emotion作为condition对生成结果的影响。

**实现代码** (`cfm.py`):
```python
# 情感dropout概率
emotion_drop_prob=0.15

# 训练时随机丢弃情感条件
drop_emotion = random() &lt; self.emotion_drop_prob
```

### 5.3 Classifier-free Guidance

**核心思想**：在生成阶段引入classifier free guidance，通过调整cfg weight调整condition对生成结果的影响，灵活调整生成语音中的情感强度。

**推理实现** (`cfm.py`):
```python
# CFG推理：同时预测条件和无条件
pred_cfg = self.transformer(
    x=x, cond=step_cond, text=text, time=t, mask=mask,
    emotion=emotion, cfg_infer=True, cache=True,
)
pred, null_pred, emo_uncond = torch.chunk(pred_cfg, 3, dim=0)
# CFG公式
return pred + (pred - null_pred) * cfg_strength + (pred - emo_uncond) * 10
```

**训练实现** (`dit.py`):
```python
# 打包条件和无条件前向传播
if cfg_infer:
    x_cond = self.get_input_embed(x, cond, text, emotion=emotion, 
                                  drop_audio_cond=False, drop_text=False, 
                                  drop_emotion=False, cache=cache)
    x_uncond = self.get_input_embed(x, cond, text, emotion=emotion,
                                    drop_audio_cond=True, drop_text=True, 
                                    drop_emotion=True, cache=cache)
    x_emo_uncond = self.get_input_embed(x, cond, text, emotion=emotion,
                                        drop_audio_cond=False, drop_text=False, 
                                        drop_emotion=True, cache=cache)
    x = torch.cat((x_cond, x_uncond, x_emo_uncond), dim=0)
```

## 六、模型架构

### 6.1 情感嵌入层

**实现代码** (`dit.py`):
```python
class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, emotion_num_embeds=None, ...):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        if emotion_num_embeds is not None:
            self.emotion_embed = nn.Embedding(emotion_num_embeds + 1, text_dim)
            nn.init.zeros_(self.emotion_embed.weight)
            self.null_emotion_id = emotion_num_embeds
    
    def forward(self, text, seq_len, emotion=None, drop_text=False, drop_emotion=False):
        text = self.text_embed(text)
        if emotion is not None and hasattr(self, "emotion_embed"):
            if drop_emotion:
                emotion = torch.full_like(emotion, self.null_emotion_id)
            text = text + self.emotion_embed(emotion)
        return text
```

### 6.2 训练配置

**中文ESD配置** (`F5TTS_ESD_CH.yaml`):
```yaml
datasets:
  name: ESD_CH
  batch_size_per_gpu: 9600
  batch_size_type: frame
  max_samples: 64

optim:
  epochs: 100
  learning_rate: 1e-5
  num_warmup_updates: 1000

model:
  name: F5TTS_v1_Base
  tokenizer: pinyin
  backbone: DiT
  arch:
    dim: 1024
    depth: 22
    heads: 16
    emotion_num_embeds: 6
```

## 七、训练与推理

### 7.1 训练流程

**训练入口** (`train.py`):
```python
@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    # 初始化模型
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, 
                             mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )
    
    # 初始化训练器
    trainer = Trainer(model, ...)
    
    # 加载数据集
    train_dataset = load_dataset(model_cfg.datasets.name, tokenizer, 
                                 mel_spec_kwargs=model_cfg.model.mel_spec)
    
    # 开始训练
    trainer.train(train_dataset, num_workers=model_cfg.datasets.num_workers)
```

### 7.2 推理接口

**推理参数** (`infer_cli.py`):
```python
parser.add_argument("--cfg_strength", type=float, 
                    help="Classifier-free guidance strength")
parser.add_argument("--nfe_step", type=int,
                    help="The number of function evaluation (denoising steps)")
```

## 八、实验结果与评估

### 8.1 训练监控

**日志记录** (`trainer.py`):
```python
if self.log_samples and self.accelerator.is_local_main_process:
    # 为每种情感生成样本
    for emo_name, emo_id in emo_map.items():
        emotion_tensor = batch["emotion"][0]
        emotion_tensor[text_len_1:] = emo_id
        emotion_tensor = emotion_tensor.unsqueeze(0)
        
        generated, _ = model_unwrapped.sample(
            cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
            text=infer_text,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            emotion=emotion_tensor,
            drop_emotion=False
        )
        
        # 保存生成的音频
        torchaudio.save(
            f"{log_samples_path}/update_{global_update}_gen_{emo_name}.wav", 
            gen_audio, target_sample_rate
        )
```

## 九、技术要点总结

### 9.1 关键创新点

1. **情感条件注入**：通过情感嵌入层将情感信息融入文本表示
2. **对比学习**：使用emotion_drop_prob实现正负样本对比
3. **CFG推理**：三路前向传播（条件、无条件、情感无条件）
4. **数据采样策略**：强制模型学习情感标签而非依赖声纹泄露

### 9.2 代码关键路径

| 功能 | 文件 |
|------|------|
| 情感嵌入 | `src/f5_tts/model/backbones/dit.py` |
| Flow Matching | `src/f5_tts/model/cfm.py` |
| CFG推理 | `src/f5_tts/model/cfm.py` |
| 数据采样 | `src/f5_tts/model/dataset.py` |
| 训练配置 | `src/f5_tts/configs/F5TTS_ESD_CH.yaml` |

## 十、未来可探索

1. 探索更复杂的情感表示方法（如情感强度、情感混合）
2. 研究更有效的声纹-情感解耦方法
3. 扩展到更多情感类别
4. 优化推理速度和生成质量
5. 在更大规模数据集上验证方法泛化性
