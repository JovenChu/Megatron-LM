1. 模型配置
   （1）首先在根目录的pretrain_gpt.py文件中定义好要训练的步骤，需要将写好的训练配置（这个在哪里？）
   （2）通过文件中的model_provider函数（P38），将配置参数传给了megatron/core/models/gpt/gpt_model.py文件的GPTModel函数
   （3）GPTModel类则继承了LanguageModule的类，构建GPT模型

3. Embedding并行
   （1）GPTModel类实例化调用了LanguageModelEmbedding函数，从而构建transformer中的embedding层
   （2）然后去构建embedding层的三种嵌入：word（词）、position（位置）、segment（句子/序列）
   （3）其中通过self.word_embeddings = tensor_parallel.VocabParallelEmbedding去构建张量的并行策略
     （a）megatron/core/tensor_parallel/layers.py中定义了并行函数VocabParallelEmbedding
     （b）
