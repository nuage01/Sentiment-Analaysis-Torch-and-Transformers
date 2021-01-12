from pandas import DataFrame
import dateparser, re, pickle
from calendar import monthrange
from dataclasses import dataclass
from typing import Any, Dict, List, Callable

PROCESSING_DICT = pickle.load(open('processing_dict.p', 'rb'))

class Compose:
    def __init__(self, transforms: List[Callable] ):
        self.transforms = transforms
        
    def __call__(self, x: Any):
        for t in self.transforms:
            x = t(x)
        return x
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
    def apply_on_df(self, df: DataFrame, col_name: str):
        df[col_name] = df[col_name].apply(lambda x: self(x))
        return df
    
@dataclass    
class Regex_Sub:
    def __call__(self, x):
        return regex_sub(x)
    
@dataclass
class Expand:        
    def __call__(self, x):
        return expand(x)

@dataclass    
class DeEmojify:        
    def __call__(self, x):
        return deEmojify(x)
    
@dataclass    
class Remove_Num:
    def __call__(self, x):
        return remove_num(x)
    
@dataclass
class Replace_Stopword:
    stopwords: List[str]           
    def __call__(self, x):
        return replace_stopword(self.stopwords, x)

@dataclass
class Remove_One_Character:
    def __call__(self, x):
        return remove_one_character(x)
    
@dataclass
class Replace_Char_With_White:
    char: str
    def __call__(self, x):
        return replace_char_with_white(self.char, x)
    
    
#############################Funtions####################################################

def replace_char_with_white(char, x):
    return x.replace(char, ' ')
    
def regex_sub(x):
    return re.sub(r'\s+', ' ', x)
    
def parse_date(x):
    partitioned = x.partition(' on ')
    
    if(partitioned[-1] == ''):
        return dateparser.parse(partitioned[0]).date()
    else:
        return dateparser.parse(partitioned[-1]).date()
    
def fix_date(d):
    d_split = d.split('-')
    year = d_split[0]
    month = d_split[1]
    day = monthrange(int(year), int(month))[1]
    return year + '-' + month + '-' + str(day)

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')

def expand(words):
    new_text = []
    for word in words.split():
        if word in PROCESSING_DICT:
            new_text.append(PROCESSING_DICT[word])
        else:
            new_text.append(word)
    return ' '.join(new_text)

def remove_num(text):
      return ''.join([i for i in text if not i.isdigit()])

def replace_stopword(word, x):
    new_x = []
    for w in x.split():
        if word != w:
            new_x.append(w)
    return ' '.join(new_x)

def remove_one_character(x):
    new_x = []
    for w in x.split():
        if len(w) != 1:
            new_x.append(w)
    return ' '.join(new_x)