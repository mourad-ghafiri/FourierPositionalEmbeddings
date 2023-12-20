import numpy as np
from sklearn.neural_network import MLPRegressor


text= """In a quiet town where whispers play
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
Wisdom old and worldly wise"""

N = 1024
t = np.linspace(0, 1, N)

vocab = sorted(list(set(text)))
int2char = {(index + 1): char for index, char in enumerate(vocab)}
char2int = {char: (index + 1) for index, char in enumerate(vocab)}
encoded = [char2int[char] for char in text]

context_size = 8
train = [encoded[i:i+context_size] for i in range(len(encoded)-context_size)]
targets = [encoded[i+context_size] for i in range(len(encoded)-context_size)]

def token_to_signal(token, position=0):
    # representation of token as a shifted square signal (inspired from spikes in the biological neural networks)
    y = np.sign(np.sin(2*np.pi*token*(t + (position/N)*np.pi)))
    return y

def context_to_signal(context):
    # Attention is calculated as sum of the shifted square signals of the tokens in the context
    signal = np.zeros(N)
    for i in range(len(context)):
        signal += token_to_signal(context[i], i)
    return signal

for i in range(len(train)):
    print(train[i], [int2char[c] for c in train[i]], targets[i], int2char[targets[i]])

X = []
Y = []
for i in range(len(train)):
    X.append(context_to_signal(train[i]))
    Y.append(token_to_signal(targets[i]))



model = MLPRegressor(
    verbose=True,
    hidden_layer_sizes=(N,) * 4, 
    solver='adam', activation="relu",
    learning_rate_init=0.001, learning_rate= "adaptive",
    batch_size=32, shuffle=True,
    max_iter=10000, tol=0.000001, n_iter_no_change=10000,
    random_state=0
)

model.fit(X, Y)


while True:
    sentence = input("Enter a sentence: ")
    for i in range(500):
        context = sentence[-context_size:]
        encoded = [char2int[char] for char in context]
        context_signal = context_to_signal(encoded)
        y = model.predict([context_signal])
        y = np.array(y).reshape(-1)
        fft = np.fft.fft(y)
        fft = np.abs(fft)[:len(fft)//2]
        prediction = np.argmax(fft)
        predictions = np.argsort(fft)[::-1][:2]
        random_choice = np.random.choice(predictions)
        try:
            # print(predictions, random_choice, int2char[random_choice])
            # print(prediction, int2char[prediction])
            sentence += int2char[prediction]
            # sentence += int2char[random_choice]
        except:
            pass
    print(sentence)
