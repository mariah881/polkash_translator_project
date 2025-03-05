#Script contains data preprocessing, tokenization, and padding functions
import spacy
import string
import torch
from torch.utils.data import DataLoader, TensorDataset
import glob

nlp = spacy.load('pl_core_news_sm')
nlp.max_length = 2643570

def define_vocab(file):
  """
  Function reads file containing a list of words for each language, converts the words to lowercase,
  deletes whitespace, ensures there are no duplicates and gets rid of ponctuation.
  Input: Path of the file containing raw vocabulary (.txt)
  Output: Clean vocabulary (.txt)
  """
  word_to_idx = {}
  with open(file, 'r', encoding='utf-8') as f:
      for i, line in enumerate(f):
          word = line.strip().lower().translate(str.maketrans("", "", string.punctuation))
          if word:
              word_to_idx[word] = i + 1

  #Adding special tokens to the vocabs
  word_to_idx["<SOS>"] = len(word_to_idx) + 1 #Start of sentence token
  word_to_idx["<EOS>"] = len(word_to_idx) + 1 #End of sentence token
  word_to_idx["<PAD>"] = len(word_to_idx) + 1 #Padding token
  return word_to_idx

#Creating special indices
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

#Function for converting sentences to indices
def sentence_to_indices(sentence, word_to_idx):
    """
    The functions converts a tokenized sentence to a list of indices, including <SOS>, <EOS>.
    """
    indexed_sentence = [word_to_idx.get("<SOS>", SOS_IDX)]
    indexed_sentence.extend([word_to_idx.get(word, word_to_idx.get("<PAD>", PAD_IDX)) for word in sentence])
    indexed_sentence.append(word_to_idx.get("<EOS>", EOS_IDX))
    return indexed_sentence

def process_files(file_paths, word_to_idx):
  """
  The function processes text files  by tokenizing and converting them into list of indices using previously defined function.
  """
  all_sentences_indices = []
  for file_path in file_paths:
    #Reading all .txt files from the corpus
      with open(file_path, 'r', encoding='utf-8') as f:
          content = f.readlines()
          for line in content:
              #Tokenization
              doc = nlp(line.strip())  #For Polish
              if word_to_idx == word_to_idx_csb:
                  doc = nlp(line.strip())  #For Kashubian

              #Tokenization and conversion to lowercase
              sentence = [token.text.lower() for token in doc]

              #Conversion of all sentences to indices according to the vocabs
              indexed_sentence = sentence_to_indices(sentence, word_to_idx)
              all_sentences_indices.append(indexed_sentence)

  return all_sentences_indices

#Padding
def pad_sentences(sentences, max_length, pad_idx=0):
    """
    The functions adds padding indices to our sentences to the given maximum length.
    It returns a list of padded sentences.
    """
    padded_sentences = []
    for sentence in sentences:
        if len(sentence) < max_length:
            #Adding padding if the sentence is shorter than max_length
            padded_sentence = sentence + [pad_idx] * (max_length - len(sentence))
        else:
            #Cutting the sentence if it is longer than max_length
            padded_sentence = sentence[:max_length]
        padded_sentences.append(padded_sentence)
    return padded_sentences

def load_data():
    ##Vocabularies: polish and kashubian
    vocab_pol_raw = r'polkash-set/pol_kash_parallel_corpus/vocabs/dictionaries_combined.pol.txt'
    vocab_csb_raw = r'polkash-set/pol_kash_parallel_corpus/vocabs/dictionaries_combined.csb.txt'

    vocab_pol = define_vocab(vocab_pol_raw)
    vocab_csb = define_vocab(vocab_csb_raw)

    word_to_idx_pol = {word: idx + 3 for idx, word in enumerate(vocab_pol)}
    word_to_idx_csb = {word: idx + 3 for idx, word in enumerate(vocab_csb)}

    #Defining variables containing sentences for each language
    polish_sentences_train_files = glob.glob(r'polkash-set/pol_kash_parallel_corpus/train/*.pol.txt')
    kashubian_sentences_train_files = glob.glob(r'polkash-set/pol_kash_parallel_corpus/train/*.csb.txt')
    polish_sentences_test_files = glob.glob(r'polkash-set/pol_kash_parallel_corpus/test/val-bleu-agnostic.pol.txt')
    kashubian_sentences_test_files = glob.glob(r'polkash-set/pol_kash_parallel_corpus/test/val-bleu-agnostic.csb.txt')
    polish_sentences_train_indices = process_files(polish_sentences_train_files, word_to_idx_pol)
    kashubian_sentences_train_indices = process_files(kashubian_sentences_train_files, word_to_idx_csb)
    polish_sentences_test_indices = process_files(polish_sentences_test_files, word_to_idx_pol)
    kashubian_sentences_test_indices = process_files(kashubian_sentences_test_files, word_to_idx_csb)

    max_length = 271 #Max sentence length

    #Padding of both Polish and Kashubian indexed sentences
    train_pol = pad_sentences(polish_sentences_train_indices, max_length)
    train_csb = pad_sentences(kashubian_sentences_train_indices, max_length)
    test_pol = pad_sentences(polish_sentences_test_indices, max_length)
    test_csb = pad_sentences(kashubian_sentences_test_indices, max_length)

    #Converting to PyTorch tensors
    train_pol_tensor = torch.tensor(train_pol, dtype=torch.long)
    train_csb_tensor = torch.tensor(train_csb, dtype=torch.long)
    test_pol_tensor = torch.tensor(test_pol, dtype=torch.long)
    test_csb_tensor = torch.tensor(test_csb, dtype=torch.long)

    #Creating DataLoaders
    train_loader = DataLoader(TensorDataset(train_pol_tensor, train_csb_tensor), batch_size=16, shuffle=True) #batch_size adjusted due to memory usage
    test_loader = DataLoader(TensorDataset(test_pol_tensor, test_csb_tensor), batch_size=16, shuffle=False)

    return train_loader, test_loader, len(word_to_idx_pol), len(word_to_idx_csb)
