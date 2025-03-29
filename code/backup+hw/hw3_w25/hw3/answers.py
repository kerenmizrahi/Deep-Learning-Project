r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    
    hypers["batch_size"] = 64
    hypers["seq_len"] = 128
    hypers["h_dim"] = 256
    hypers["n_layers"] = 2  # omer did 4
    hypers["dropout"] = 0.1
    hypers["learn_rate"] = 0.001
    hypers["lr_sched_factor"] = 0.08  #omer did 0.5
    hypers["lr_sched_patience"] = 0.6  #omer did 5
    
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    
    #start_seq = "ACT I. \n\nSCENE I. King's palace. \n\nEnter KING, and TOM. \n\nKING: Now, sir, the news that I have heard of you."
    start_seq = "ACT I. \n\nSCENE I. King's palace. \n\nEnter KING and REGAN. \n\nREGAN: My lord, I beg you to give me the opportunity"
    temperature = 0.9
    
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

First, training the model on the entire text would require large tensors that could exeed memory limits. Training on shorter sequences makes the training process feasible. Additionally, spliting the curpus into smaller sequences can increase efficiency, as shorter sequences leed to faster gradient calculations and may allow for more parallel computations, since multiple sequences can be processed independetly.

"""

part1_q2 = r"""
**Your answer:**

The ability to generate text with memory longer than the sequence length comes from how RNNs maintain and update their hidden state during processing. As the RNN processes each token in the sequence, it updates its hidden state, which serves as a form of 'memory' of the past tokens. This hidden state allows the model to retain important contextual information, enabling it to remember details from earlier parts of the sequence, resulting in contextually relevant output even beyond the sequence's length.

"""

part1_q3 = r"""
**Your answer:**

We do not shuffle the order of batches when training an RNN model for text generation because the data is sequential. Shuffling would disrupt the natural order of the data, preventing the model from retaining context and learning long-term dependencies. As a result, the model could make inaccurate predictions by relying on misleading context.

"""

part1_q4 = r"""
**Your answer:**

1. Lowering the temperature for sampling makes the probability distribution less uniform (which is used for selecting the next character). A less uniform distribution reduces randomness, making the predictions more deterministic. This results in fewer options for the model to choose from and increases the likelihood of selecting the most probable character.

2. When the temperature is very high, the probability distribution becomes more uniform and 'flatter.' This reduces the differences between probabilities, bringing less likely and more likely characters closer in likelihood. As a result, the model's predictions become more random and less deterministic.

3. When the temperature is very low, the probability distribution becomes more peaked around the most likely characters. This makes the model consistently choose the most probable character at each timestep, which can result in overly predictable text or repetitive loops.

"""
# ==============


# ==============
# Part 2 answers

#PART2_CUSTOM_DATA_URL = None
PART2_CUSTOM_DATA_URL = "https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip"

def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=64,
        h_dim=512,
        z_dim=256,
        x_sigma2=0.0004,
        learn_rate=0.0002,
        betas=(0.9, 0.999),
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The $\sigma^2$ hyperparam represents the variance of the likelihood function, controlling how closely the model fits the training data.

$\bullet$ Low $\sigma^2$ - The model focuses on precise reconstruction, resulting in a sharper images but potentially overfitting the training data. Reduces the diversity in generated samples.

$\bullet$ High $\sigma^2$ - The model allows more variation, leading to a diverse but blurrier outputs. Acts as regularization which prevents overfitting but sacrifices details.

"""

part2_q2 = r"""
**Your answer:**

1.

$\bullet$ Reconstruction Loss - Measures the difference between the reconstructed sample and the original input using the L2 norm. It ensures the model generates accurate and realistic outputs. A lower reconstruction loss indicates better reconstructions.

$\bullet$ KL Divergence Loss - Acts as a regularization term, measuring how much the learned latent distribution deviates from the standard normal prior $N(0,1)$. A lower KL loss encourages a structured and continuous latent space, preventing overfitting and improving generalization.

2.

Effect on Latent-Space Distribution - The KL loss encourages the model to shape the latent space closer to a normal distribution. This ensures that nearby points in the latent space produce similar outputs, preventing clustering and gaps.

3.

Benefit of This Effect - A well-structured latent space enables smooth interpolation between samples, allowing meaningful transitions between generated images. This improves diversity, generalization, and the overall quality of generated outputs.


"""

part2_q3 = r"""
**Your answer:**

We maximize the distribution $p(X)$ because it ensures that the model learns to generate realistic data while maintaining a structured latent space.

$\bullet$ Upper Bound Approximation - Since directly computing $p(X)$ is intractable, we use the ELBO as a surrogate objective to approximate it.

$\bullet$ Neglecting KL Divergence of $q(Z|x) || p(Z|x)$ - We assume this term is relatively small compared to $p(X)$, allowing us to ignore it and simplify optimization.

Which enables efficient training while ensuring that the learned distribution is as close as possible to the true data distribution.
"""

part2_q4 = r"""
**Your answer:**

We model $\log \sigma^2_{\alpha}$ instead of $\sigma^2_{\alpha}$ because it simplifies the computation of the log-likelihood when using the L2 norm.

$\bullet$ Efficient Computation - Since the VAE loss involves a Gaussian log-likelihood, modeling $\log \sigma^2_{\alpha}$ directly aligns naturally with the objective function.

$\bullet$ Numerical Stability - Directly modeling $\sigma^2_{\alpha}$ can lead to exploding gradients or extremely small values, making training unstable. Using the log transformation helps avoid these issues.

$\bullet$ Prevent Negative Variance - The variance must be positive, and by learning $\log \sigma^2_{\alpha}$, it ensure us that the final variance value $\sigma^2_{\alpha} = \exp(\log \sigma^2_{\alpha})$ is always positive without needing additional constraints.
"""

# Part 3 answers
'''
# --- ORIGINAL ---
def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.1
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.5, 0.999)
    # ========================
    return hypers
'''
#---------------------------- FROM REF UNTIL THEY ANS AT PIAZZA --------------------------
def part3_gan_hyperparams():
    return {
        'batch_size': 32,  
        'z_dim': 100,       
        'discriminator_optimizer': {
            'type': 'Adam',
            'lr': 0.0001,   
            'betas': (0.5, 0.999)  
        },
        'generator_optimizer': {
            'type': 'Adam',
            'lr': 0.00015,  
            'betas': (0.5, 0.999)  
        },
        'data_label': 1,
        'label_noise': 0.05 
    }

part3_q1 = r"""
**Your answer:**


"""

part3_q2 = r"""
**Your answer:**


"""

part3_q3 = r"""
**Your answer:**



"""

#PART3_CUSTOM_DATA_URL = None
PART3_CUSTOM_DATA_URL = "https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip"

def part4_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        dropout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers = dict(
        embed_dim = 128,
        num_heads = 4,
        num_layers = 3,
        hidden_dim = 200,
        window_size = 32,
        dropout = 0.2,
        lr = 0.0004,
    )
    # ========================
    return hypers




part4_q1 = r"""
**Your answer:**

Staching encoder layers that use the sliding-window attention results in a broader context in the final layer due to information propogation.

$\bullet$ **Single Layer** - each token initally only attends to a fixed window size, restricting  direct interactions.

$\bullet$ **Multi Layer** - tokens indirectly receive information from beyond their immidiate window because the neighbouring tokens pass along their attended information.

$\bullet$ **CNN Analogy** - stacking small conv kernels increases the receptive field, stacking sliding-window attention layers allows each token to access a wider reange of sequences.
"""

part4_q2 = r"""
**Your answer:**

here are a few possible solutions:

$\bullet$ **Dilated Sliding Window** - attend to every $i^{th}$ neigbour instead of to $k^{th}$ closest neighbours. This expands the receptive field of each token without increasing the number of attended tokens, which allows information to propagate faster across the sequence while maintaining $O(nw)$ complexity.

$\bullet$ **Strided Global Attention** - every $k^{th}$ token is designated as a global token which attends all tokens in the sequence while the rest of the tokens still use sliding-window attention, which allows key tokens to propagate info across the entire sequence while keeping most of the computations local. $O(nw)$ complexity.


"""


part5_q1 = r"""
**Your answer:**

The results clearly show that fine-tuning a pre-trained model significantly outpreforms training it from scratch:
1. **Fine-tuned model**

    $\bullet$ **2 Frozen layers:** Train Accuracy 76%, Test Accuracy 77%

    $\bullet$ **Fully fine-tuned:** Train Accuracy 82%, Test Accuracy 87%

2. **Trained-from-scratch model**

    $\bullet$ **Trained-from-scratch model:** Train Accuracy 65.8%, Test Accuracy 65.7%

The Fine-Tuned model preformed better because:

$\bullet$ It started with a rich language representation learned from massive text corpora. it already understood word relationships, context and sentence structure, requiring only minor task-specific tuning, while the trained-from-scratch model had to learn everything from random initialization, making learning much harder.

$\bullet$ It built on a pre-trained foundation, l eading to faster training and better generalization with limited data, while the trained-from-scratch model is prone to overfitting or underfitting because it lacks strong priors.

This will not always be the case for any downstream task, for example:

$\bullet$ The downstream task is very diffrent making the pre-trained model not as helpful.

$\bullet$ The pre-trained model is already specific to our task, making the fine tuning not as helpful.
"""

part5_q2 = r"""
**Your answer:**

If we froze the last layers and fine-tuned internal layers such as the multi-headed attention block, the model's performance would likely **be worse** compared to the standard fine-tuning approach because of reasons such as:

$\bullet$ The last layers are responsible for task-specific classification. In our case, they are trained to map the learned embeddings into positive/negative labels. If these layers remain frozen, the model cannot properly adapt to the new IMDb dataset.

$\bullet$ The multi-headed attention block mainly captures contextual word relationships, it does not directly determine the final classification output. Fine-tuning these layers without adjusting the classifier limits the model's ability to make task-specific decisions.

In conclusion, The model **may still improve slightly** because internal fine-tuning can refine word representations, however, since the classifier is frozen, the model cannot fully adjust to IMDb sentiment classification, leading to not-so optimal accuracy.
"""


part5_q3= r"""
**Your answer:**


"""

part5_q4 = r"""
**Your answer:**

The main reasons to choose RNN over Transformers are:

$\bullet$ **Memory Efficiency:** RNNs process input sequentially, meaning that they require less memory at any given time compared to Transformers, which store attention matrices that grow a lot!

$\bullet$ **Streaming Applications:** RNNs process data step ata time - making them good for real-time\streaming taks where the entire input sequence is not available at once.

$\bullet$ **Short Sequences:** If the input sequences are already short, the benefits of self-attention in Transformers become less relevant, making RNN preform similarly while being computationally simpler.

$\bullet$ **Low-resource Environments:** RNNs can be lighter computationally making them useful for deployment on low-power devices.

"""

part5_q5 = r"""
**Your answer:**

NSP is a binary classification task used during BERT's pre-training to help the model understand sentence relationships.

$\bullet$ The model receives two sentences: $A$ and $B$.

$\bullet$ With 50% probability, $B$ is the real next sentence that follows $A$ in the original corpus.

$\bullet$ In the remaining 50% of cases, sentence $B$ is a random sentence from another part of the dataset.

**Where the prediction occurs:** The **[CLS] token** from the input is used to make the binary classification decision: "Is B the actual next sentence or not?".

**The Loss function:** A binary cross-entropy loss is used for this classification task.

**Why NSP might be useful:**

$\bullet$ **Understanding sentence relationships:** helps BERT learn coherence and logical structure between sentences.

$\bullet$ **Better contextual embeddings:** enourages BERT to model longer range dependencides, mking the representations be more informative

**Why NSP isn't essential:**

$\bullet$ **Was Replaced in later models:** some later models removed NSP entirely and still achieve better results, suggeesting that NSP isn't as crucial.
"""


# ==============
