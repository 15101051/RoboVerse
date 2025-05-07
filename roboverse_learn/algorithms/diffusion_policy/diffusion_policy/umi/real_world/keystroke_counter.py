import time
from pynput.keyboard import Key, KeyCode, Listener
from collections import defaultdict
from threading import Lock

class KeystrokeCounter(Listener):
    def __init__(self):
        self.key_count_map = defaultdict(lambda:0)
        self.key_press_list = list()
        self.key_press_time_list = list()
        self.lock = Lock()
        super().__init__(on_press=self.on_press, on_release=self.on_release)
    
    def on_press(self, key):
        press_time = time.time()
        with self.lock:
            self.key_count_map[key] += 1
            self.key_press_list.append(key)
            self.key_press_time_list.append(press_time)
    
    def on_release(self, key):
        pass
    
    def clear(self):
        with self.lock:
            self.key_count_map = defaultdict(lambda:0)
            self.key_press_list = list()
            self.key_press_time_list = list()
    
    def __getitem__(self, key):
        with self.lock:
            return self.key_count_map[key]
    
    def get_press_events(self):
        with self.lock:
            events = list(self.key_press_list)
            self.key_press_list = list()
            self.key_press_time_list = list()
            return events
        
    def get_press_events_with_time(self):
        with self.lock:
            events = list(zip(self.key_press_list, self.key_press_time_list))
            self.key_press_list = list()
            self.key_press_time_list = list()
            return events

if __name__ == '__main__':
    import time
    with KeystrokeCounter() as counter:
        try:
            while True:
                print('Space:', counter[Key.space])
                print('q:', counter[KeyCode(char='q')])
                time.sleep(1/60)
        except KeyboardInterrupt:
            events = counter.get_press_events()
            print(events)
