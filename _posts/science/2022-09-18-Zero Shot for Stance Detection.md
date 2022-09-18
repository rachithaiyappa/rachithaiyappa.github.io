---
layout: post
title: "Zero Shot for Stance Detection"
categories: science
tags: Research NLP
share : true
comments : true
---


TLDR - Zero Shot inference outperforms the **best** performers of SemEval 2016 in the stance detection task!


<img src="/images/NLP_ZeroShot.png" alt="NLP" class="center">

I spent the last three months learning about Large Language Models (LLMs). Well, I've been hearing about them since *forever*, but I never sat down to learn what goes on "under the hood." There is so much to write about, especially about the evolution of Natural Language Processing (NLP) but that will be for a different blog post. Fun fact, the picture above is generated using [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release)!

Here, I'm going to try and do justice to an insane capability of LLM --- zero shot stance detection. 

Think about a tweet: `Vaccines are harmful. Please stay away from it.`   
For those of you on twitter, it is not hard to imagine coming across such a tweet. Many different fields of research, like computational social science, are interested in inferring the *stance* to *targets* in such tweets (why? well, why do you breathe?). That is, is the tweet in `favour` of `vaccination` or is it `against` it? 
Here `favour/against` is the potential stance towards the target, `vaccination.` 

*Yes, I still use favour over favor.*

A conventional machine learning method to infer stance is to collect and label a bunch of data and then train a classifier to predict the stance, based on some features of the data. 
For example, here is some made up data and their stance towards `vaccination`, 
1. Vaccines can kill you. --- against
2. Please do not get the vaccine. It has grave side effects. --- against
3. Vaccines saved my life. --- favour

Using such labelled data, one can define and create a vector of features of the data, like the presence of negative (kill, grave, etc.) or positive (saved, etc.) words. This can then be used to infer the stance of the tweet by asking, "which of the two categories of words is more prominent in the selected tweet?" 

It should be evident that this process is, 
1. a cumbersome: did you say manual annotation? Goodbye. 
2. flawed: Consider the example, `I am sad that I am not eligible to get the vaccine.` Here, `sad` is a negative word which might tempt a classifier to categorise the stance towards vaccination as negative, which is obviously incorrect. This example, also shows the difference between sentiment analysis and stance detection. The tweet has a negative sentiment but supports vaccinations.  Ofcourse there are [more sophisticated feature selections methods to address this](https://arxiv.org/abs/2006.03644). 
3. soooo pre 2017.

So how can we approach this problem today? Enter LLMs. But wait, what are language models? I'm glad you asked and I'm sorry this is not the post for it. You have google for that. 

To keep this blog simple, you would not be wrong if you think of a LLM as a human *(aha! caught red-handed saying AI has dominated us)*. A good (whatever that means) LM can be asked to fill in the blank in ``A tiger is _____`` and it (ze?) **can** rightly tell you `an animal` . Ask a question, `What is the capital of India?` and you could get `Delhi`. I'm tempted to write out all the nuances here, but no...

More interestingly, LLMs, unlike featured-based machine learning pipelines, can be trained, in a semi-supervised manner without a specific downstream task in mind (called `pretraining`). That is, they can be trained on a simple task like predicting a random missing word in a sentence picked from, say, a (digital) book or wikipedia. Pretrained models can then be finetuned using **fewer** labelled data pertaining to the task in hand, which in this blog post, is stance detection. While this pretraining-finetuning pipeline, eliminates the need for feature engineering and does better at the task (thanks to `transfer learning`), it still relies on labelled data. 

However, LLMs have yet another, relatively under-explored capability --- zero shot inference. [Here](https://joeddav.github.io/blog/2020/05/29/ZSL.html) is a cool introduction to what zero shot means. Briefly, instead of finetuning a pretrained LM, one can directly make a pretrained LM perform a desired task! No labelled data required whatsoever. 

Wait, what!? Yup. 

For stance detection, one can use a LM like [BART](https://arxiv.org/abs/1910.13461) to perform a natural language inference (NLI) task. Particularly, feed BART a `premise` which is a tweet whose stance is to be inferred and a `hypothesis` which is a prompt indicating a possible stance and ask BART to infer if the premise entails the hypothesis.  
Technically, BART returns a score indicating the probability $p$ that the premise entails the hypothesis ($1-p$ is then the probability that the premise contradicts the hypothesis). This process can then be repeated for a number of stance prompts. The stance of the tweet is then the stance of the prompt belonging with the maximum entailment score. 

For the premise mentioned above, `Vaccines are harmful. Please stay away from it.`, two hypotheses can be `This statement is against vaccination` and `This statement is in favour of vaccination`. Try out each of them in this [hugging face space I created](https://huggingface.co/spaces/rachith/ZeroShot_StanceDetection)and let BART tell you which of these is the stance of the tweet! Cool innit? 

Well, sure it might work out for this cherry picked example, but is it really *that* good?

The table below is a snippet of a result from some experiments I performed. I used a dataset from [SemEval 2016](https://aclanthology.org/S16-1003/) which basically encompasses a number of tweets and a target pertaining to that tweet (can loosely be interpreted as the topic of the tweet). The goal is to infer the stance towards the target. The table shows the performance of zero shot inference compared to the best performing models of the SemEval 2016...a strong baseline to have! Note, the best model submitted to the competition, differed across targets. 

Meaning of the columns:  
1. Target and Stance - The `stance` (favour or against) expressed by the tweet towards the `target`. Note the target need not be explicitly mentioned in the tweet! 
2. $\bar{F}_{zs}$ - Average of the F scores obtained by zero shot inference on the favor and against task.  
3.  $\bar{F}_{se}$ - Average of the F scores obtained by the best performing model (submitted to the SemEval 2016 competition) for a selected target on the favor and against task.  
4. The rank of zero-shot inference among approximately 30 submissions.  

*Drumroll....*

Zero shot inference using BART, **without having seen any training data** related to the task is able to outperform or come close to the best performing models which were trained on for the task! This should make you go :O

Most interesting is the last row of the table which is the stance towards Donald Trump. This particular target was a struggle for the competitors since it did not have any training data available. This tests the generalisability of the models trained on labelled data pertaining to the other targets. Clearly, this is no problem for zero-shot inference. Why should it be!? It does not need training data anyway! Duh...


| Target      | Stance | $\bar{F}_{zs}$ | $\bar{F}_{se}$| Zero-shot rank |
| :---        |   :----: |  :----:   |  ---: |  :---: |
| Legalisation of Abortion      | Favour<br>Against       | 0.57 | 0.66 | 12 |
| Hillary Clinton      | Favour<br>Against       | 0.75 | 0.67 | 1 |
| Feminist Movement      | Favour<br>Against       | 0.65 | 0.62 | 1 |
| Climate Change is a real concern   | Favour<br>Against |  0.43 | 0.54 | 4 |
| Atheism      | Favour<br>Against       | 0.75 | 0.67| 1    |
| Donald Trump      | Favour<br>Against       | 0.78 | 0.56| 1   |


You might have a bunch of questions. Well, good news...my friends and I have them too and are actively pursuing this! You should join us :D
