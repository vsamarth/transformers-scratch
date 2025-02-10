# %%
import seaborn as sns
import jax

sns.set_theme()

# %%
# Read the dataset

words = open("names.txt").read().splitlines()
print(f"Using {len(words)} names")
words[:10]

# %%
# Explore the dataset
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

M = len(stoi)
print(M, itos)

# %%
import jax.numpy as jnp

context_length = 3

X = []
y = []
for w in words:
    context = [0] * context_length
    for ch in w + ".":
        idx = stoi[ch]
        X.append(context)
        y.append(idx)
        context = context[1:] + [idx]


X = jnp.array(X)
y = jnp.array(y)

X.shape, y.shape

jax.default_backend()

# %%
from jax import random, Array, jit, vmap, value_and_grad
from jax.nn import one_hot, softmax
import jax

#  Define the model
token_space = 27
embedding_space = 2
key = random.key(42)
key, C_key, W1_key, W2_key = random.split(key, 4)
parameters = {
    "C": random.normal(C_key, (token_space, embedding_space)),
    "W1": random.normal(W1_key, (embedding_space * context_length, 100)) * 0.1,
    "W2": random.normal(W2_key, (100, token_space)) * 0.2,
}


@jit
def model(X: Array, parameters: dict[str, Array]):
    emb = jnp.dot(one_hot(X, token_space), parameters["C"]).reshape(
        context_length * embedding_space
    )
    hlogits = jnp.tanh(jnp.dot(emb, parameters["W1"]))
    logits = jnp.dot(hlogits, parameters["W2"])
    probs = softmax(logits)
    return probs


@jit
def criterion(probs: Array, y: int):
    return -jnp.log(probs[y])

# %%
-jnp.log(1/27)

# %%
@value_and_grad
def forward(parameters: dict[str, Array], X: Array, y: Array):
    preds = vmap(model, in_axes=(0, None))(X, parameters)
    loss = jnp.mean(vmap(criterion)(preds, y))
    return loss

for i in range(2000):
    loss, grad = forward(parameters, X, y)
    for k in parameters.keys():
        parameters[k] -= .1 * grad[k]
    if (i % 100 == 0):
        print(f"Epoch {i}: {loss}")




