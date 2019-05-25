"""By: Xiaochi (George) Li: github.com/XC-Li"""
import pandas as pd
import os
from util_code.xml_parser import bs_parser, xml_parser, get_person_speech_pair
# from xiaodan.data_loader import get_data
from tqdm.autonotebook import tqdm  # auto backend selection


# get xml data info: This function is written by Xiaodan Chen
def get_data(path):
    """
    get one specific xml path
    """
    all_path = []

    def searchPath(path):
        for item in os.listdir(path):
            subFile = path + "/" + item
            if os.path.isdir(subFile):
                searchPath(subFile)
            else:
                if subFile.split('.')[-1] == "xml":  # bug fix: out of range
                    # get all path
                    # path = subFile
                    all_path.append(subFile)

    searchPath(path)
    return all_path


def corpus_loader(debug=False, parser='bs', data_root='../opinion_mining/'):
    """
    Corpus Loader: Match the record between action.csv and document.csv and load corpus from XML
    Args:
        debug(Bool): the switch for debug
        parser(str): which parser to use, bs/xml
        data_root(str): the root of data and labels
    Returns:
        Pandas DataFrame
    """
    # data_root = '../opinion_mining/'
    corpus_root = data_root + 'cr_corpus/'
    action = pd.read_csv(data_root + 'action_data.csv')
    document = pd.read_csv(data_root + 'document_data.csv')

    count = match = no_match = 0
    data_list = []
    no_match_list = []
    for index, _ in action.iterrows():
        count += 1
        cr_pages = action.loc[index, 'cr_pages']
        support = 1 if action.loc[index, 'action_text_string'].startswith('Support') else -1
        person_id = action.loc[index, 'person_id']
        cr_date = action.loc[index, 'cr_date'][:10]
        #     print(cr_pages, support, person_id, cr_date)
        first_name = action.loc[index, 'first_name']
        middle_name = action.loc[index, 'middle_name']
        last_name = action.loc[index, 'last_name']
        full_name = first_name + '-' + str(middle_name) + '-' + last_name
        party = action.loc[index, 'party_abbreviation']
        chamber = action.loc[index, 'chamber']
        title = action.loc[index, 'brief_title']
        doc = document.loc[(document['page_range'] == cr_pages) & (document['pub_date'] == cr_date)]
        if len(doc) == 0:
            if debug:
                print('No match', cr_pages, support, person_id, cr_date)
            # doc = document.loc[(document['page_range'].str.contains(cr_pages)) & (document['pub_date'] == cr_date)]
            # if len(doc) == 0:
            #     print('still no match')

            # no_match_list.append([cr_pages, support, person_id, cr_date])
            no_match += 1
            continue
        volume_no = doc['volume_no'].iloc[0]
        issue_no = doc['issue_no'].iloc[0]
        page_range = doc['page_range'].iloc[0]
        #     print(volume_no, issue_no, page_range)
        path = corpus_root + str(volume_no) + '/' + str(issue_no) + '/' + str(page_range) + '/'
        for file in os.listdir(path):
            # print(i,':', person_id, ':',path+file)
            if parser == 'bs':
                text = bs_parser(path + file, person_id)
            else:
                text = xml_parser(path + file, person_id)
            if len(text) > 0:
                # print('match')
                match += 1
                data_list.append([support, text, volume_no, issue_no, page_range,
                                  person_id, full_name, party, chamber, title])

    column_name = ['support', 'text', 'volume_no', 'issue_no', 'page_range',
                   'person_id', 'full_name', 'party', 'chamber', 'title']
    data_frame = pd.DataFrame(data_list)
    data_frame.columns = column_name
    print('Total:', count, 'Match:', match, 'No Match:', no_match)
    # no_match_df = pd.DataFrame(no_match_list)
    # no_match_df.columns = ['cr_pages', 'support', 'person_id', 'cr_date']
    # no_match_df.to_csv('no_match.csv')
    return data_frame


def untagged_corpus_loader(tagged_df=None, path_root='../opinion_mining'):
    """
    untagged corpus loader: Load all the untagged corpus from XML files
    Args:
        tagged_df(Pandas DataFrame): the tagged data frame
        path_root(str): the root of the path to search all XML files
    Returns:
        untagged_data_frame(Pandas DataFrame): untagged data frame, in the same format of tagged_df
    """
    # tagged_df = corpus_loader()
    if tagged_df is not None:
        tagged_ids = tagged_df['volume_no'].map(str) + '/' + tagged_df['issue_no'].map(str) + '/' \
                     + tagged_df['page_range'].map(str) + ':' + tagged_df['person_id'].map(str)
    else:
        tagged_ids = []
    all_xml_path = get_data(path_root)
    # print(len(all_xml_path))

    untagged_data_list = []
    total = untagged = 0
    for file_name in tqdm(all_xml_path):
        total += 1
        volume_no = file_name.split('/')[-4]
        issue_no = file_name.split('/')[-3]
        page_range = file_name.split('/')[-2]
        person_id_speech_pair = get_person_speech_pair(file_name)

        for person_id in person_id_speech_pair:
            unique_id = volume_no + '/' + issue_no + '/' + page_range + ':' + person_id
            if unique_id not in tagged_ids:
                untagged += 1
                text = person_id_speech_pair[person_id]
                support = full_name = party = chamber = title = 0
                untagged_data_list.append([support, text, volume_no, issue_no, page_range,
                                           person_id, full_name, party, chamber, title])

    column_name = ['support', 'text', 'volume_no', 'issue_no', 'page_range',
                   'person_id', 'full_name', 'party', 'chamber', 'title']
    untagged_data_frame = pd.DataFrame(untagged_data_list)
    untagged_data_frame.columns = column_name
    print('Total files processed:', total, 'Total untagged speeches:', untagged)
    return untagged_data_frame


if __name__ == '__main__':
    tagged_df = corpus_loader()
    untagged_df = untagged_corpus_loader(tagged_df)
