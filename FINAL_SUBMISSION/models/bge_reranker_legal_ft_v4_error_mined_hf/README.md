---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:988
- loss:BinaryCrossEntropyLoss
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model trained using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'XLMRobertaForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Kocanın iznin kaldırılması bakımından mahkeme veya hâkim hangi hususları dikkate almalıdır?', 'ORICON\n\nÖngörülen bir aylık süre hak düşürücü süre olup mahkemece resen dikkate alınacaktır.'],
    ['Yürütmenin durdurulması kararı bakımından dava süreci veya başvuru koşulları nasıl açıklanmaktadır?', 'ORICON\nİdare ve İdari Yargılama Hukuku\nYürütmenin durdurulması kararı için itiraz süresi yedi gündür ve bu süre kararın tebliğini izleyen günden itibaren başlar. Yürütmenin durdurulması istemi neticesinde verilen karara karşı bir defalığına mahsus itiraz edilmesi mümkündür; bu durum, itiraz üzerine verilen kararın kesin olmasından kaynaklanmaktadır. Yetkili merci, itiraz üzerine yürütmenin durdurulması istemini yedi gün içinde karara bağlamak zorundadır. İYUK m. 27/10 hükmü gereği, aynı sebeplere dayanarak ikinci kez yürütmenin durdurulması isteminde bulunulması mümkün değildir. Yürütmenin durdurulması kararı ile davanın nihai olarak reddedilmesi arasındaki sürede, yürütmenin durdurulması kararı iptal kararı gibi sonuç doğurmaktadır. İYUK madde 28/1 gereği, yürütmenin durdurulması kararı verildikten sonra idare, bu karar doğrultusunda gecikmeksizin işlem tesis etmeye veya eylemde bulunmaya mecburdur.'],
    ['Kamulaştırma Bedelinin Tespiti; Kısmi Kamulaştırma uyuşmazlığında kararın temel hukuki değerlendirmesi nedir?', 'YARGITAY_HGK_RETRIEVAL_LOW_RISK_ONLY\nHukuk Genel Kurulu 2013/2298 E., 2015/1028 K. | Kamulaştırma Bedelinin Tespiti; Kısmi Kamulaştırma\nBu nedenle, değerlendirmede esas alınan kiraz ve şeftalinin dekara verim miktarı ve 2011 yılı toptan kilogram satış fiyatı ile üretim giderleri resmi kuruluşlardan getirtilip, bilirkişi kurulundan bu yönde ek rapor alınması gerektiği düşünülmeden, 2010 yılına göre değer belirleyen raporun hükme esas alınması, 2)Dava konusu taşınmazın kamulaştırmadan arta kalan ve bilirkişi raporunda A harfi ile gösterilen kısmın geometrik durumu ve yüzölçümü gözetilerek bu bölüme %50 oranında değer kaybı verilmesi gerektiği gözetilmeden, az bedel tespiti, 3)4650 sayılı Kanunla değişik 2942 sayılı Kamulaştırma Kanununun 10.maddesine dayanan kamulaştırma bedelinin tespiti davalarında, aynı Yasanın 25/son maddesi uyarınca mülkiyet idareye tescille geçeceğinden, ağaçların değerinden enkaz bedelinin indirilmeyeceğinin düşünülmemesi, 4)Resmi kuruluşların yargı harcından bağışık tutulabilmesi için özel kanunlarında yargı harcından muaf olduğunun açıkça belirtilmesi gerekir. Diğer harçlardan bağışık tutulma yargı harcını kapsamaz. Bu nedenle, davacı idareden harç alınması gerekirken, harç alınmasına yer olmadığına karar verilmesi, Doğru görülmemiştir...) gerekçesiyle hükmün bozulmasına karar verilerek dosya yerine geri çevrilmekle, yeniden yapılan yargılama sonunda, mahkemece önceki kararda direnilmiştir. TEMYİZ EDEN: Davalı vekili'],
    ['Derneklerin zorunlu organları genel kurul konusunda hangi şartlar, yükümlülükler veya hukuki sonuçlar açıklanmaktadır?', 'ORICON\nGenel hukuk / sınıflandırma bekliyor\nDernek tüzüğünde derneğin adı, amacı, gelir kaynakları, üyelik koşulları, organları, örgütü ve geçici yönetim kurulunun gösterilmesi zorunludur.'],
    ['Kişi özgürlüğü ve güvenliği hakkı - tutuklama, gözaltı, tahliye iddiası bakımından bireysel başvuruda hangi gerekçe öne çıkmaktadır?', 'TRAIN_LOW_RISK_ONLY\nTRAIN.CSV AYM Karar Özeti - KAYIT 0327\nKAYIT 0327 | Karar sonucu: İhlal | Temel hak konusu: Kişi özgürlüğü ve güvenliği hakkı | Uyuşmazlık alanı: Ceza yargılaması | Ayırt edici kavramlar: tutukluluk, tutuklama, tutuklu, gözaltı. Olaylar: başvuru formu ve eklerine göre olaylar özetle şöyledir. Başvurucu cumhuriyet başsavcılığınca yürütülen sayılı soruşturma kapsamında tarihinde gözaltına alınmış ağır ceza mahkemesinin tarihli ve sorgu sayılı kararıyla suç işlemek amacıyla kurulan örgüte üye olma suç örgütüne yarar sağlamak amacıyla yağma öldürme ve ruhsatsız ateşli silah taşıma suçlarından tutuklanmıştır ağır ceza mahkemesi tarihli duruşmada başvurucunun tutukluluk hâlinin devamına karar vermiştir. Başvurucu bu karara karşı tarihinde itiraz yoluna başvurmuştur. Başvurucu tarihinde bireysel başvuruda bulunmuştur tarihli duruşmada verilen tutukluluk hâlinin devamına ilişkin karara. Başvurucu tarafından yapılan itiraz ağır ceza mahkemesinin tarihli ve değişik sayılı kararıyla reddedilmiştir ağır ceza mahkemesi tarihli ve sayılı kararıyla başvurucunun üzerine atılı suçlardan mahkûmiyetine ve hükümle birlikte tutukluluk hâlinin devamına karar vermiştir anılan karar temyiz edilmiş olup temyiz incelemesi hâlen devam etmektedir b hukuk tarihli ve sayılı ceza muhakemesi kanununun veya sanığın salıverilme istemleri kenar başlıklı maddesinin numaralı fıkrası şöyledir ceza mahkemesinin görevine giren işlerde tutukluluk süresi en çok iki yıldır bu süre zorunlu hallerde gerekçesi gösterilerek uzatılabilir uzatma süresi toplam üç yılı geçemez sayılı maddesi şöyledir...'],
]
scores = model.predict(pairs)
print(scores)
# [1.0071e-04 4.0606e-04 3.4599e-04 9.5355e-05 4.3393e-04]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Kocanın iznin kaldırılması bakımından mahkeme veya hâkim hangi hususları dikkate almalıdır?',
    [
        'ORICON\n\nÖngörülen bir aylık süre hak düşürücü süre olup mahkemece resen dikkate alınacaktır.',
        'ORICON\nİdare ve İdari Yargılama Hukuku\nYürütmenin durdurulması kararı için itiraz süresi yedi gündür ve bu süre kararın tebliğini izleyen günden itibaren başlar. Yürütmenin durdurulması istemi neticesinde verilen karara karşı bir defalığına mahsus itiraz edilmesi mümkündür; bu durum, itiraz üzerine verilen kararın kesin olmasından kaynaklanmaktadır. Yetkili merci, itiraz üzerine yürütmenin durdurulması istemini yedi gün içinde karara bağlamak zorundadır. İYUK m. 27/10 hükmü gereği, aynı sebeplere dayanarak ikinci kez yürütmenin durdurulması isteminde bulunulması mümkün değildir. Yürütmenin durdurulması kararı ile davanın nihai olarak reddedilmesi arasındaki sürede, yürütmenin durdurulması kararı iptal kararı gibi sonuç doğurmaktadır. İYUK madde 28/1 gereği, yürütmenin durdurulması kararı verildikten sonra idare, bu karar doğrultusunda gecikmeksizin işlem tesis etmeye veya eylemde bulunmaya mecburdur.',
        'YARGITAY_HGK_RETRIEVAL_LOW_RISK_ONLY\nHukuk Genel Kurulu 2013/2298 E., 2015/1028 K. | Kamulaştırma Bedelinin Tespiti; Kısmi Kamulaştırma\nBu nedenle, değerlendirmede esas alınan kiraz ve şeftalinin dekara verim miktarı ve 2011 yılı toptan kilogram satış fiyatı ile üretim giderleri resmi kuruluşlardan getirtilip, bilirkişi kurulundan bu yönde ek rapor alınması gerektiği düşünülmeden, 2010 yılına göre değer belirleyen raporun hükme esas alınması, 2)Dava konusu taşınmazın kamulaştırmadan arta kalan ve bilirkişi raporunda A harfi ile gösterilen kısmın geometrik durumu ve yüzölçümü gözetilerek bu bölüme %50 oranında değer kaybı verilmesi gerektiği gözetilmeden, az bedel tespiti, 3)4650 sayılı Kanunla değişik 2942 sayılı Kamulaştırma Kanununun 10.maddesine dayanan kamulaştırma bedelinin tespiti davalarında, aynı Yasanın 25/son maddesi uyarınca mülkiyet idareye tescille geçeceğinden, ağaçların değerinden enkaz bedelinin indirilmeyeceğinin düşünülmemesi, 4)Resmi kuruluşların yargı harcından bağışık tutulabilmesi için özel kanunlarında yargı harcından muaf olduğunun açıkça belirtilmesi gerekir. Diğer harçlardan bağışık tutulma yargı harcını kapsamaz. Bu nedenle, davacı idareden harç alınması gerekirken, harç alınmasına yer olmadığına karar verilmesi, Doğru görülmemiştir...) gerekçesiyle hükmün bozulmasına karar verilerek dosya yerine geri çevrilmekle, yeniden yapılan yargılama sonunda, mahkemece önceki kararda direnilmiştir. TEMYİZ EDEN: Davalı vekili',
        'ORICON\nGenel hukuk / sınıflandırma bekliyor\nDernek tüzüğünde derneğin adı, amacı, gelir kaynakları, üyelik koşulları, organları, örgütü ve geçici yönetim kurulunun gösterilmesi zorunludur.',
        'TRAIN_LOW_RISK_ONLY\nTRAIN.CSV AYM Karar Özeti - KAYIT 0327\nKAYIT 0327 | Karar sonucu: İhlal | Temel hak konusu: Kişi özgürlüğü ve güvenliği hakkı | Uyuşmazlık alanı: Ceza yargılaması | Ayırt edici kavramlar: tutukluluk, tutuklama, tutuklu, gözaltı. Olaylar: başvuru formu ve eklerine göre olaylar özetle şöyledir. Başvurucu cumhuriyet başsavcılığınca yürütülen sayılı soruşturma kapsamında tarihinde gözaltına alınmış ağır ceza mahkemesinin tarihli ve sorgu sayılı kararıyla suç işlemek amacıyla kurulan örgüte üye olma suç örgütüne yarar sağlamak amacıyla yağma öldürme ve ruhsatsız ateşli silah taşıma suçlarından tutuklanmıştır ağır ceza mahkemesi tarihli duruşmada başvurucunun tutukluluk hâlinin devamına karar vermiştir. Başvurucu bu karara karşı tarihinde itiraz yoluna başvurmuştur. Başvurucu tarihinde bireysel başvuruda bulunmuştur tarihli duruşmada verilen tutukluluk hâlinin devamına ilişkin karara. Başvurucu tarafından yapılan itiraz ağır ceza mahkemesinin tarihli ve değişik sayılı kararıyla reddedilmiştir ağır ceza mahkemesi tarihli ve sayılı kararıyla başvurucunun üzerine atılı suçlardan mahkûmiyetine ve hükümle birlikte tutukluluk hâlinin devamına karar vermiştir anılan karar temyiz edilmiş olup temyiz incelemesi hâlen devam etmektedir b hukuk tarihli ve sayılı ceza muhakemesi kanununun veya sanığın salıverilme istemleri kenar başlıklı maddesinin numaralı fıkrası şöyledir ceza mahkemesinin görevine giren işlerde tutukluluk süresi en çok iki yıldır bu süre zorunlu hallerde gerekçesi gösterilerek uzatılabilir uzatma süresi toplam üç yılı geçemez sayılı maddesi şöyledir...',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 988 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 988 samples:
  |         | sentence_0                                                                        | sentence_1                                                                           | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                               | float                                                          |
  | details | <ul><li>min: 16 tokens</li><li>mean: 29.3 tokens</li><li>max: 73 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 216.91 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.08</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                 | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:---------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Kocanın iznin kaldırılması bakımından mahkeme veya hâkim hangi hususları dikkate almalıdır?</code>                   | <code>ORICON<br><br>Öngörülen bir aylık süre hak düşürücü süre olup mahkemece resen dikkate alınacaktır.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | <code>0.0</code> |
  | <code>Yürütmenin durdurulması kararı bakımından dava süreci veya başvuru koşulları nasıl açıklanmaktadır?</code>           | <code>ORICON<br>İdare ve İdari Yargılama Hukuku<br>Yürütmenin durdurulması kararı için itiraz süresi yedi gündür ve bu süre kararın tebliğini izleyen günden itibaren başlar. Yürütmenin durdurulması istemi neticesinde verilen karara karşı bir defalığına mahsus itiraz edilmesi mümkündür; bu durum, itiraz üzerine verilen kararın kesin olmasından kaynaklanmaktadır. Yetkili merci, itiraz üzerine yürütmenin durdurulması istemini yedi gün içinde karara bağlamak zorundadır. İYUK m. 27/10 hükmü gereği, aynı sebeplere dayanarak ikinci kez yürütmenin durdurulması isteminde bulunulması mümkün değildir. Yürütmenin durdurulması kararı ile davanın nihai olarak reddedilmesi arasındaki sürede, yürütmenin durdurulması kararı iptal kararı gibi sonuç doğurmaktadır. İYUK madde 28/1 gereği, yürütmenin durdurulması kararı verildikten sonra idare, bu karar doğrultusunda gecikmeksizin işlem tesis etmeye veya eylemde bulunmaya mecburdur.</code>                                                                                     | <code>0.0</code> |
  | <code>Kamulaştırma Bedelinin Tespiti; Kısmi Kamulaştırma uyuşmazlığında kararın temel hukuki değerlendirmesi nedir?</code> | <code>YARGITAY_HGK_RETRIEVAL_LOW_RISK_ONLY
  Hukuk Genel Kurulu 2013/2298 E., 2015/1028 K. | Kamulaştırma Bedelinin Tespiti; Kısmi Kamulaştırma
  Bu nedenle, değerlendirmede esas alınan kiraz ve şeftalinin dekara verim miktarı ve 2011 yılı toptan kilogram satış fiyatı ile üretim giderleri resmi kuruluşlardan getirtilip, bilirkişi kurulundan bu yönde ek rapor alınması gerektiği düşünülmeden, 2010 yılına göre değer belirleyen raporun hükme esas alınması, 2)Dava konusu taşınmazın kamulaştırmadan arta kalan ve bilirkişi raporunda A harfi ile gösterilen kısmın geometrik durumu ve yüzölçümü gözetilerek bu bölüme %50 oranında değer kaybı verilmesi gerektiği gözetilmeden, az bedel tespiti, 3)4650 sayılı Kanunla değişik 2942 sayılı Kamulaştırma Kanununun 10.maddesine dayanan kamulaştırma bedelinin tespiti davalarında, aynı Yasanın 25/son maddesi uyarınca mülkiyet idareye tescille geçeceğinden, ağaçların değerinden enkaz bedelinin indirilmeyeceğinin düşünülmemesi, 4)Resmi kuruluşların yargı harcından bağı...</code> | <code>0.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 1
- `per_device_eval_batch_size`: 1
- `num_train_epochs`: 1
- `fp16`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `do_predict`: False
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 1
- `per_device_eval_batch_size`: 1
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: None
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `enable_jit_checkpoint`: False
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `use_cpu`: False
- `seed`: 42
- `data_seed`: None
- `bf16`: False
- `fp16`: True
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: -1
- `ddp_backend`: None
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `auto_find_batch_size`: False
- `full_determinism`: False
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `use_cache`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.5061 | 500  | 0.4727        |


### Training Time
- **Training**: 5.6 minutes

### Framework Versions
- Python: 3.12.13
- Sentence Transformers: 5.4.1
- Transformers: 5.0.0
- PyTorch: 2.10.0+cu128
- Accelerate: 1.13.0
- Datasets: 4.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->