# Recurrent Neural Networks

## Diagram

<figure markdown="span">
  ![Image title](assets/rnn.png){ width="400" }
  <figcaption>Diagram of a RNN</figcaption>
</figure>

- Input: $\textbf{x} \in \texttt{R}^{d}$  (i.e. word of a sentence)
- Hidden state: $\textbf{s} \in \texttt{R}^{m}$, where $m$ is the number of hidden units
- Output: $\textbf{o} \in \texttt{R}^{n}$ (i.e. next word of the sentence)
- Parameters: $\textbf{U} \in \texttt{R}^{m \times d}$, $\textbf{V} \in \texttt{R}^{n \times m}$, $\textbf{W} \in \texttt{R}^{m \times m}$ (they are the same for each time step $t$)


## Cell definition

$$
\textbf{s}_{t} = \sigma (\textbf{W}\textbf{s}_{t-1} + \textbf{U}\textbf{x}_{t} + \textbf{b})
$$

$$
\textbf{o}_{t} = \sigma (\textbf{V}\textbf{s}_{t} + \textbf{c})
$$

## Training

