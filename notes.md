SI 630 Lecture notes: Training RNNs


"Old school" ~ 2014 Era Neural Networks
* Recurrent Neural Networks:
	* Need to deal with sequences of different lengths
	* RNN allow for conditioning on arbitrarily long input sequences
	* Recall:
		* Seek to build RNN using hidden state to capture previous context 
		* Has some sort of "Memory gate" of what the previous state was at time $t-1$
			* More or less allows for the ability to remember what came previously in the sequence of inputs
		* RNN Process:
			* State is dependenton 
				* $X_t$: input at step $t$
				* $O_t$: Output at step $t$
				* $S_t$: Hidden state at step $t$
            * Key distinction: Backpropagation now becomes backpropagation through time, taking the derivative of the loss incurred at step t wrt parameters we want to update

            $$ \frac{\partial L(\theta)_{y1}}{\partial W^s} \frac{\partial L(\theta)_{y2}}{\partial W^s} \cdots \frac{\partial L(\theta)_{yt}}{\partial W^s}$$
            * Extenstion: Can translate to parts of speech (POS) tagging
        * Bidirectional RNN: Has both a forward and backward RNN that traverse the sequence of input to parse
            * Needs to be in an appropriate application space
            * Think Forward / Backward Algorithm https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
            * Incredibly sequential for computation

        * Long short-term memory (LSTM)
            * Concept: SPlit s vector propagated between time steps into a memory component tand a hidden state component
            * Designed to address vanishing gradient problem that arises when you have long input sequences (numerical underflow)
    * Theory for Homework 2: Using word embeddings for document classification (recap)
        * Use max, weighted sum, etc. (some aggregation function) to assign importance to words
        * Solution: Learn weighting for aggregation with an attention layer
        * Attention matrix:
        $$v \in \mathbb{R}^d$$
        $$ a = softmax(r)$$
    * Different variations on attention:
        - Linear transformation of x
        - non-linearities after each operation
        - multiheaded attention
        - Etc.
* Attention in Networks: Given a sequence of words, how can we leverage attention for prediction?
    - Sequence to Sequence or seq2seq models (aka encoder decoder model)
    - Generate next word conditional on previous word and hidden state
    - W size is |vocab| x |hidden state|, softmax output over entire vocab
    - Decoder has separate parametersfrom the encoder, so this can learn to be a language model (produce next plausible word from previous input)
    - Issue: Tends to get stuck in loops (decoder never bothers to decode the 'STOP' token)
        - Need to remedy with some notion of input coverage and what words have already been translated
        - Bad at processing long sentences:
            - Fixed size hidden representation doesn't scale
            - LSTMs struggle to remember for long period of time (large degrees of separation are problematic)
        - Other issues:
            - Encoding rare words into a vector space is difficult
            - Aligned input: if we know in advance that the source and target words have 1:1 correspondence
            - Approach: Attend to specific input tokens when decoding the next output

- Neural Machine Translation (NMT)
    - Encoder / Decoder approach: RNN passes output to other RNNs
    - Think sequence of RNNs, chained together
    - Attention Mechanism combines all attention vectors into a context vector that then is taken as input to compute the next hidden state (concatenate context vector with the current hidden state)
        - Think of context vector as similar to HW2, where it is a weighted sum of attention vectors on the encoder side
- Mathematics of attention:
    - Without attention: $P(y_i|x, y_1, \cdots , y_{i-1}) = softmax(W^\intercal h_i)$
    - With attention: $P(y_i|x, y_1, \cdots , y_{i-1}) = softmax(W^\intercal [c ; h_i])$ where $c_i = \sum_j \alpha_{ij} h_j$
    - $\alpha_{ij} = softmax(e_{ij})$
    - $e_{ij} = f(\bar{h}_i, h_j)$ for some function $f$
    - possible functions for $f$:
        - hyperbolic tangent
        - dot product
        - bilinear $\bar{h}_i^\intercal W h_j$

    - Encoder hidden states capture word context / representation
- Transformer Architecture
    - Attention is all you need: Created new architecture for dealing with sequences
    - Combine pairs of words witha sum and then compute a function fromthe term embeddings T x T matrix @ T x D term embedding matrix for token T in input sequence (D is size of embedding)
    - Output looks uniform because every word is equally important (if you have a matrix of ones)
        - Identity matrix would simply return word embedding values
        - Need to learn the attention matrix via neural net
        - May or may not be symmetric matrix? Interesting
        - Input sequence can be arbitrarily long
        - Output matrix represents the new embedding for the word based on its importance to all other words
    - Alternate view: Create a matrix that captures how important word i is to word j
        - T by D queries matrix Q
        - Keys.T: T by D matrix K 
        -  Q @ K = Weights Matrix W
    - Size of values in the matrix can get arbitrarily large; need normalize
        - Values: embeddings of original word before considering context
        - Queries = some representation that you want to compute attention for
        - Keys: All the other words in the sequence
        - Formula: $\frac{Q @ K^\intercal}{\sqrt{D}}$
        - Softmax(W,dim=1) @ Values = Attention modified vectors: TxD matrix
        - q, k, v are just linear layers in pytorch
        - See proof for how the Formula controls for variance in the weight matrix
- Unlike LSTM models, the encode is a feedforward network - no recurrence!
    - Most of this efficiency comes from the self attention embedding in the encoder
    - Can stack encoders on top of one another
    - Allows model to learn how relevant each word is to every other word
- Keep track of where words are with positional embedding
- Each position in the sentence gets a unique vector that is added to the input embedding
- Creates constant offset (some vector) that modifies the word embeddings to indicate that the word is located at a given location
- Random Coding Models: Orthogonal vectors in some high dimensional space

- Transformer Architecture significantly outperformed the prevailing architectures at the time
    - Now, nearly every NLP model uses some form of BERT or these pre-trained models
    - Lots of good reading (see slides)

- Future of NLP: Pretrained Models
    - Inspiration came from computer vision and imagenet
        - Imagenet: collection of many labeled items
        - Hierarchical structure helps for inferring meaning
        - Discovery: Pretrained classification model to learn what images look like before adapting parameters to specific tasks in CNNs
            - Neurons learn higher order / level information 
        - Significant increase in performance from the pretraining
    - Text based pretraining: Use BERT
        - Apply same concept from image classification to text processing
        - Masked language modeling and next-sentence prediction to learn parameters of BERT layers (trained on Wikipedia + BookCorpus)
        - Then add new linear transformation + softmax to get distribution over output space (trained on annotated data)
    - Bert Tasks: Two fold
        - Task 1: Infilling 
            - Randomly replace (mask) words from the input and have BERT try to predict missing word
        - Second Task: Identify "fake" sentence pairs to try to separate real from fake
            - Ex. I went to the MASK and bought MASK gallon of dog. I love Karaoke!
    - Iterate between these two tasks and have BERT learn word representations; loss function is softmax
    - NOT a generative model; however, learns really effective representations: See "BERT has a mouth, let it speak" paper
        - Key distinction from word2vec: BERT learns contextual word representations
        - At the end, we have one representation for each layer for each token
        - Moving up in layers, we see a diffusion of word meanings across different tokens

- Additional Aspects of BERT:
    - CLS token: some sort of contextualized information that represents all words in the sentence
    - [SEP] token: Are these two sentences (current and next) related?
- Probing Bert: Attempting to determine what each layer is responding to
    - learn mixing weights $h_{i, \tau} = \gamma_\tau \sum_{l = 0}^L s_\tau^{(l) \mathbf{h}_i^{(l)}}$ of word piece i for task $\tau$
    - At least 60% of BERT's heads can be removed with only a minimal drop in performance
    - DistilBert

- Other language models:
    - GPT: Generative language model, always trying to predict the last word
    - Causal language models: Review slides

- Masked Language Models:
    - Class of models that look at forward context to fill in the blank
    - Includes BERT

- Causal Language Models: 
    - More like previously encountered models
    - Can only look at past context to predict future words
    - Generally used for language generation

- Collection of these models known as "Foundation Models"
    - The training cost is incredibly time intensive and costly (order of $100K each, just to train the final version of the model)


    



$ \lambda $
