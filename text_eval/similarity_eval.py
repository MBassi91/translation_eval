from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import nltk
from nltk.translate.bleu_score import sentence_bleu


"""
A very simple function to calculate similariy beteween two texts using BERT embeddings.
"""

# Initialize BERT tokenizer and model
#tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
#model = BertModel.from_pretrained('bert-base-multilingual-cased')

#NLTK's resources downloaded for BLEU score calculation
nltk.download('punkt')

# Function to calculate BERT embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(1)  # Mean pooling
    return embeddings.detach().numpy()

# Function to calculate similarity based on BERT
def calculate_bert_similarity(text1, text2):
    embedding1 = get_embeddings(text1)
    embedding2 = get_embeddings(text2)
    return cosine_similarity(embedding1, embedding2)[0][0]

# Function to calculate BLEU score
def calculate_bleu_score(reference, candidate):
    # NLTK expects the reference as a list of sentences and each sentence as a list of words
    # Similarly, the candidate sentence should be tokenized into a list of words
    reference_tokens = [nltk.word_tokenize(reference)]
    candidate_tokens = nltk.word_tokenize(candidate)
    score = sentence_bleu(reference_tokens, candidate_tokens)
    return score

# Function to choose between BERT and BLEU for similarity calculation
def calculate_similarity(text1, text2, method='bert'):
    if method == 'bert':
        return calculate_bert_similarity(text1, text2)
    elif method == 'bleu':
        return calculate_bleu_score(text1, text2)
    else:
        raise ValueError("Unsupported method. Choose either 'bert' or 'bleu'.")

# Example usage
text1 = "This is a sample sentence. I like it very much. It is very good."
text2 = "This is a similar sentence. I enjoy it a lot. It is very good."

print(calculate_similarity(text1, text2, method='bleu'))