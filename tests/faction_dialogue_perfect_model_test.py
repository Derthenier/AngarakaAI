import sys, os
sys.path.append(os.getcwd())

from models.dialogue.train_perfect_model import PerfectDialogueInference

inference = PerfectDialogueInference()

tests = [
    ('What is karma?', 'ashvattha'),
    ('Can karma be measured?', 'vaikuntha'), 
    ('How do we break free from suffering?', 'yuga_striders'),
    ('Is reality just a story?', 'shroud_mantra')
]

for question, faction in tests:
    response = inference.generate_response(question, faction)
    print(f'{faction}: {response}')
    print()