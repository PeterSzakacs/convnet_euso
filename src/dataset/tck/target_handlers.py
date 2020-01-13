import dataset.constants as cons


HANDLER_TYPES = ('STATIC', 'BIN_COLUMN')


def get_target_handler(handler_type, **kwargs):
    _type = handler_type.upper()
    if _type == 'STATIC':
        return StaticTargetHandler(kwargs['target_value'])
    elif _type == 'BIN_COLUMN':
        return BinaryColumnTargetHandler(kwargs['column_name'])
    else:
        raise ValueError(f'Unknown handler type: {handler_type}')


class StaticTargetHandler:

    def __init__(self, target_value):
        self.target_value = target_value

    def process_events(self, events):
        target = self.target_value
        for event_list in events:
            yield [(event[0], target, event[1]) for event in event_list]


class BinaryColumnTargetHandler:

    def __init__(self, column_name):
        self.column_name = column_name

    def process_events(self, events):
        column = self.column_name
        targets = (cons.CLASSIFICATION_TARGETS['noise'],
                   cons.CLASSIFICATION_TARGETS['shower'])
        for event_list in events:
            # event[0] - extracted packet
            # event[1] - dict with added metadata incl. target column value
            yield [(event[0], targets[int(float(event[1][column]))], event[1])
                   for event in event_list]
