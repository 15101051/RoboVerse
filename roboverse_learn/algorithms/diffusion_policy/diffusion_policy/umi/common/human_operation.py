from abc import ABC
import numpy as np


class HumanOperation(ABC):
    def __init__(self, type):
        self.type = type
    
    @classmethod
    def interpolate_human_operation(cls, timestamps, human_operation_list):
        def all_signal():
            for i in range(len(human_operation_list)):
                if human_operation_list[i].type != 'signal':
                    return False
                return True
        if all_signal():
            res, cur_ope = [], 0
            for ti in range(len(timestamps)):
                if cur_ope < len(human_operation_list):
                    assert ti + 1 < len(timestamps), f'timestamps:{timestamps}\nhuman_operation_list:{human_operation_list}'
                    if timestamps[ti + 1] > human_operation_list[cur_ope].signal_time:
                        res.append(human_operation_list[cur_ope].signal_type)
                        cur_ope += 1
                    else:
                        res.append(chr(0))
                else:
                    res.append(chr(0))
            assert cur_ope == len(human_operation_list)
            assert len(res) == len(timestamps)
            return np.array(res)
        else:
            raise NotImplementedError


class HumanSignal(HumanOperation):
    def __init__(self, signal_time, signal_type: str):
        super().__init__('signal')
        self.signal_time = signal_time
        self.signal_type = signal_type

    def __repr__(self):
        return f'HumanSignal(signal_time:{self.signal_time}, signal_type:{self.signal_type})'
    
    def __str__(self):
        return f'HumanSignal(signal_time:{self.signal_time}, signal_type:{self.signal_type})'
        