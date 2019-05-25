"""By: Xiaochi (George) Li: github.com/XC-Li"""
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from bs4 import BeautifulSoup


def bs_parser(file_name, target_id):
    """
    XML Parser implemented by Beautiful Soup Package
        Args:
        file_name(str): path to the document
        target_id(str): the person_id of speaker of target document
    Returns:
        speech(str): the speech of the speaker
    """
    text_list = []
    with open(file_name, encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'xml')
        target_speech = soup.find_all('speaker', personId=target_id)
        if len(target_speech) > 1:
            pass
            # print('multiple speech:', target_id, file_name)
        for item in target_speech:
            # s = item.get_text(strip=False)  # this will cause the string in subtag concatenated together
            for s in item.stripped_strings:  # bug fix: fix the problem on previous line
                text_list.append(s)
    return ' '.join(text_list)


def xml_parser(file_name, target_id):
    """
    XML Parser implemented by xml package
    Args:
        file_name(str): path to the document
        target_id(str): the person_id of speaker of target document
    Returns:
        speech(str): the speech of the speaker
    """
    try:
        tree = ET.parse(file_name, ET.XMLParser(encoding='utf-8'))
        root = tree.getroot()
    except ParseError:
        with open(file_name, encoding='utf-8') as temp:
            file_data = temp.read()
            file_data = file_data.replace('&', 'and')
            root = ET.fromstring(file_data)

    text_list = []
    for child in root[0]:
        if child.tag == 'speaker':
            if 'personId' in child.attrib:  # contain person ID
                person_id = child.attrib['personId']
            else:
                continue
            if str(person_id) != str(target_id):  # multiple speaker in a document, not target speaker
                continue
            for item in child.findall('p'):
                if len(item) == 0:
                    text_list.append(item.text)
                else:  # multiple sub tag inside 'p' tag
                    if item.text is not None:
                        text_list.append(item.text)
                        text_list.append(' ')
                    for i in item:
                        if i.text is not None:
                            text_list.append(i.text)
                            text_list.append(' ')
                        if i.tail is not None:
                            text_list.append(i.tail)
                            text_list.append(' ')

    return ''.join(text_list)


def xml_to_person_id(file_name):
    """
    NO LONGER USEFUL
    XML parser to get the person_ids from given XML file
    Args:
        file_name(str): file name
    Returns:
        person_ids(set[int]): a set of person ids
    """
    person_ids = set()
    with open(file_name, encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'xml')
        all_speech = soup.find_all('speaker')
        for single_speech in all_speech:
            try:
                person_ids.add(single_speech['personId'])
            except KeyError:
                continue
    return person_ids


def get_person_speech_pair(file_name):
    """
    XML parser to get the person_ids from given XML file
    Args:
        file_name(str): file name
    Returns:
        person_id_speech_pair(dict): Dict[person_id(int) -> speech(str)]
    """
    person_id_speech_dict = dict()
    with open(file_name, encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'xml')
        all_speech = soup.find_all('speaker')
        for single_speech in all_speech:
            try:  # newer format
                person_id = single_speech['personId']
            except KeyError:
                try:  # older format
                    person_id = single_speech['person']
                except KeyError:
                    continue
            single_speech_list = []
            for s in single_speech.stripped_strings:
                single_speech_list.append(s)
            processed_speech = ' '.join(single_speech_list)
            #             print(parsed_speech, '\n')
            if person_id not in person_id_speech_dict:
                person_id_speech_dict[person_id] = []
            person_id_speech_dict[person_id].append(processed_speech)

    for person_id in person_id_speech_dict:
        person_id_speech_dict[person_id] = ' '.join(person_id_speech_dict[person_id])
    return person_id_speech_dict


if __name__ == '__main__':
    # Sample: multiple sub-tag inside p tag
    # print(xml_parser('../opinion_mining/cr_corpus/160/8/E63-E64/4407923.xml', '404'))
    # print(bs_parser('../opinion_mining/cr_corpus/160/8/E63-E64/4407923.xml', '404'))
    print(get_person_speech_pair("D:\\Github\\FN-Research-GW-Opinion-Mining\\opinion_mining\\cr_corpus\\146\\4\\E8-E10\\27718.xml"))
