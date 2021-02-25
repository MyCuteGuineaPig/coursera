#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 14:50:56 2021

@author: beckswu
"""

import numpy as np
from w2v_utils import *

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

def cosine_similarity(u, v):
    return np.dot(u.T, v)/(np.linalg.norm(u)*np.linalg.norm(v))


father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____. 
    
    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    best_word = None
    max_cosine_sim = float("-inf")
    for word in words:
        if word in [word_a, word_b, word_c]:
            continue
        diff = cosine_similarity(word_to_vec_map[word_a]-word_to_vec_map[word_b],
                   word_to_vec_map[word_c] - word_to_vec_map[word])
        if diff > max_cosine_sim:
            max_cosine_sim = diff
            best_word = word
            
    return best_word


triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))
    
    
g = word_to_vec_map['woman'] - word_to_vec_map['man']
print(g)

print ('List of names and their similarities with constructed vector:')
# girls and boys name
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
    
    
    
def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
    This function ensures that gender neutral words are zero in the gender subspace.
    
    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.
    
    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """   
    
    gmatrix = np.matrix(g).reshape((50,1))
    #if not convert to matrix
    #np.dot(g, g.T)  is scaler
    # np.dot(gmatrix, gmatrix.T) is 50 x 50 matrix
    proj1 = np.dot( np.dot(gmatrix, gmatrix.T) / np.linalg.norm(g)**2, word_to_vec_map[word])
    
    proj2 = np.dot(g.T, word_to_vec_map[word]) / np.linalg.norm(g)**2 * g
    
    
    
    e_debiased = word_to_vec_map[word]- np.array(proj1).reshape(50,)
    #e_debiased = word_to_vec_map[word]- proj2
    return e_debiased

e = "receptionist"
print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))
    


def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.
    
    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors
    
    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """
    
    w1, w2 = pair
    ew1, ew2 = word_to_vec_map[w1], word_to_vec_map[w2]


    mu = np.mean(np.array([ew1, ew2]),axis=0)
    mu_B = np.dot(mu, bias_axis)/np.linalg.norm(bias_axis)**2 * bias_axis
    mu_orth = mu - mu_B
    
    ew1B = np.dot(ew1, bias_axis)/np.linalg.norm(bias_axis)**2 * bias_axis
    ew2B = np.dot(ew2, bias_axis)/np.linalg.norm(bias_axis)**2 * bias_axis
    
    corrected_e_w1B = (np.sqrt(abs(1 - np.linalg.norm(mu_orth)**2)) * (ew1B - mu_B) / np.linalg.norm(ew1 - mu))
    corrected_e_w2B = (np.sqrt(abs(1 - np.linalg.norm(mu_orth)**2)) * (ew2B - mu_B) / np.linalg.norm(ew2 - mu))

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections (≈2 lines)
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth
                                                                
    
    return e1, e2
    
print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))