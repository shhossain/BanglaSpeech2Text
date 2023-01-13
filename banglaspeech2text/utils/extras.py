from hashlib import md5
import os
from collections import defaultdict
from threading import Thread
from banglaspeech2text.utils import get_app_path, app_name
import pickle

class ShortTermMemory:
    def __init__(self, size):
        self.size = size
        self.memory = defaultdict(int)
        self.memory_keys = []
        self.cache_file = os.path.join(get_app_path(app_name), "cache.pkl")
        
        if os.path.exists(self.cache_file):
            self.load(self.cache_file)
    
    def add(self, key, value):
        if len(self.memory) >= self.size:
            del self.memory[self.memory_keys.pop(0)]
        
        self.memory[key] = value
        self.memory_keys.append(key)    
        self.save()
        
    def __save(self, file_path):
        with open(file_path, "wb") as f:
            data = (self.memory, self.memory_keys)
            pickle.dump(data, f)
            
    def save(self):
        t = Thread(target=self.__save, args=(self.cache_file,))
        t.start()
    
    def load(self, file_path):
        with open(file_path, "rb") as f:
            try:
                data = pickle.load(f)
            except EOFError:
                data = (defaultdict(int), [])
            self.memory, self.memory_keys = data
            
    def clear(self):
        self.memory = defaultdict(int)
        self.memory_keys = []
        
    
    def get(self, key):
        return self.memory.get(key)
    
    def __contains__(self, key):
        return self.memory[key] != 0
    
    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, key):
        return self.memory[key]
    
    def __setitem__(self, key, value):
        self.add(key, value)
        
    def __delitem__(self, key):
        del self.memory[key]
        self.memory_keys.remove(key)
    
    
        
def get_hash(file_path):
    # get first and last pars of file
    with open(file_path, "rb") as f:
        first = f.read(1024)
        f.seek(-1024, os.SEEK_END)
        last = f.read(1024)

    # get md5 hash of first and last pars
    first_hash = md5(first).hexdigest()
    last_hash = md5(last).hexdigest()

    # get md5 hash of first and last hash
    return md5((first_hash + last_hash).encode()).hexdigest()

    
    

    