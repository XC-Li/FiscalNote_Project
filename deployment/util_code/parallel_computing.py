"""By: Xiaochi (George) Li: github.com/XC-Li"""
import multiprocessing
from multiprocessing import freeze_support
from util_code.preprocess_utility import remove_stopwords
from tqdm import tqdm_notebook as tqdm
from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize

def f(x):
    return x * x


def test_main():
    freeze_support()
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    xs = list(range(5))
    result = []
    for y in tqdm(pool.imap(f, xs), total=len(xs)):
        result.append(y)          # 0, 1, 4, 9, 16, respectively
    return result


def parallel_remove_stopwords(x):
    freeze_support()
    text_list = x.tolist()
    cores = multiprocessing.cpu_count()
    chunk = int(len(text_list) / cores)

    pool = multiprocessing.Pool(processes=cores)
    result = []
    for proceed_text in pool.imap(remove_stopwords, text_list, chunksize=chunk):
        result.append(proceed_text)
    return result


from preprocess_utility import remove_stopwords_return_list


# def tag_document(in_):
#     i = in_[0]
#     d = in_[1]
#     result = TaggedDocument(words=remove_stopwords_return_list(d, 500), tags=[str(i)])
#     return result


# def parallel_load_gensim(pd_series):
#     freeze_support()
#     data = pd_series.tolist()
#     tagged_data = []
#     cores = multiprocessing.cpu_count()
#     pool = multiprocessing.Pool(processes=cores)
#     chunk = int(len(data) / cores)
#     for single_tagged in pool.imap(tag_document, enumerate(data), chunksize=chunk):
#     # for single_data in tqdm(enumerate(data)):  # unparallel version
#     #     single_tagged = tag_document(single_data)
#         tagged_data.append(single_tagged)
#     return tagged_data


if __name__ =='__main__':
    test_main()