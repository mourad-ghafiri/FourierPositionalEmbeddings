import torch
import torch.nn as nn
import torch.nn.functional as F

d_model = 1024


class MyModel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.w1 = nn.Linear(d_model, d_model)

    def forward(self, inputs):
        t = torch.linspace(0, 1, self.d_model).requires_grad_(False)
        signals = []
        signal = torch.zeros(self.d_model)
        for position, input in enumerate(inputs):
            p = torch.log(1 + torch.tensor(position).requires_grad_(False))
            signals.append(torch.sin(2 * torch.pi * input * (t + p)))
            signal += torch.sin(2 * torch.pi * input * (t + p))
        signal = signal.requires_grad_(False)
        x = self.w1(signal)
        return x


text= """
In a quiet town where whispers play
Lives a creature night and day
A feline spirit soft and sly
Underneath the moonlit sky
With eyes like orbs of gleaming gold
Stories untold ancient and old
Paws that tread on silent ground
In their steps a mystery found
Whiskers twitch in the gentle breeze
Dancing lightly among the trees
Ears that listen to the night's song
In a world where they belong
Fur as soft as the morning's dew
In shades of black white or blue
They roam the streets without a care
Grace in each step light as air
In gardens lush and fields wide
Their elegant forms do glide
Masters of the shadow's dance
In their gaze you're caught in trance
By day they bask in sunlit beams
In slumber deep chasing dreams
Of mice that scamper in their play
In the realm of night and day
In ancient times they were revered
In pyramids their forms appeared
Guardians of the secrets old
In their eyes the stories told
In alleyways and on the fence
Their mystery makes perfect sense
A creature both wild and tame
Never twice quite the same
They purr like the rolling sea
A sound of peace and mystery
A lullaby for troubled hearts
In their presence warmth imparts
With agile leap and graceful bound
They traverse their hallowed ground
In every movement there's a poem
In every silence a hidden tome
In winter's chill or summer's heat
Their resilience is quite a feat
Adapting with such ease and grace
In every season they find their place
Some say they have nine lives to live
In each one love they freely give
Teachers of the art of being
In their gaze a deeper seeing
In their eyes a galaxy spins
A universe where wonder begins
Each whisker a line of a verse
In their world no need for rehearse
They play with yarn in sheer delight
In their joy the world turns bright
Chasing shadows pouncing on light
In their games a pure delight
At times they seem to ponder deep
Secrets in their hearts they keep
Sages in a furry guise
Wisdom old and worldly wise
"""

text = text.lower()
tokens = text.split(" ")
vocab = sorted(list(set(tokens)))
int2char = {(index + 1): char for index, char in enumerate(vocab)}
char2int = {char: (index + 1) for index, char in enumerate(vocab)}
encoded = [char2int[char] for char in tokens]

context_size = 4
train = [encoded[i:i+context_size] for i in range(len(encoded)-context_size)]
targets = [encoded[i+context_size] for i in range(len(encoded)-context_size)]

for item in range(len(train)):
    print(f"{' '.join([int2char[c] for c in train[item]])} {train[item]} -> {targets[item]} {int2char[targets[item]]}")


t = torch.linspace(0, 1, d_model).requires_grad_(False)
targets = [torch.sin(2*torch.pi*torch.tensor(target)*t) for target in targets]


model = MyModel(d_model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


for epoch in range(100):
    for i in range(len(train)):
        y = model(train[i])
        target = targets[i]
        loss = criterion(y, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} | Loss {loss.item()}")


while True:
    sentence = input("Enter a sentence: ")
    for i in range(300):
        context = sentence.lower()
        tokens = context.split(" ")
        context = tokens[-context_size:]
        encoded = [char2int[char] for char in context]
        y = model(encoded)
        fft = torch.fft.fft(y)
        fft = torch.abs(fft)[:len(fft)//2]
        prob = F.softmax(fft*0.5, dim=0)
        prediction = torch.multinomial(prob, num_samples=1).item()
        # print(prediction)
        # print(prediction, int2char[prediction])
        sentence += (" " + int2char[prediction])
    print(sentence)
