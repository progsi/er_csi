from src.dataset import OnlineCoverSongDataset
from src.utils import Config
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from torch.optim import Adam
from sentence_transformers import models, SentenceTransformer


#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

config = Config("sbert", "shs100k2_train", "shs100k2_val", 100, 16)


# init training iterator
dataset_train = OnlineCoverSongDataset(
    config.train_dataset_name,
    config.data_path,
    config.yt_metadata_file)
    
    
# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(dataset_train, shuffle=True, batch_size=1)

# Define the model
word_embedding_model = models.Transformer('bert-base-uncased')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Define the optimizer and loss
optimizer = Adam(model.parameters(), lr=1e-6)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), 
                                num_labels=2)

# Training loop
for epoch in range(2):  # adjust as needed
    for batch in train_dataloader:
        ref_sentence, positive_example, negative_examples = batch
        
        # FIXME: this is ugly and needs to be fixed
        ref_sentence = ref_sentence["yt_title"][0]
        positive_example = positive_example["yt_title"][0]
        negative_examples = [s["yt_title"][0] for s in negative_examples]
        
        # Forward pass
        ref_embedding = model.encode(ref_sentence)
        positive_embedding = model.encode(positive_example)
        negative_embeddings = [model.encode(neg_example) for neg_example in negative_examples]

        # Compute loss
        loss = train_loss(ref_embedding, positive_embedding, negative_embeddings)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the trained model
model.save('sbert_model')