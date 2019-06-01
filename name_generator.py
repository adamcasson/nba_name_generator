#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:19:37 2019

@author: adam
"""

import numpy as np
from basketball_reference_web_scraper import client

def get_player_names(season_end_years=[2019]):
    """Gets a list of player first names and last names for the
        for the list of seasons given
        
    Args:
        season_end_years (list, optional): A list of seasons to retrieve
            player names for. The end year of the season is expected.
            i.e. for the 2018-2019 season, the method expects just 2019
            
    Returns:
        list: two lists, one for the first names and one for the last 
            names
    
    """
    # Query basketball reference to get player season total stats for 
    # the given years.
    player_totals = []
    for year in season_end_years:
        player_totals.extend(client.players_season_totals(season_end_year=year))
        
    # Collects only the names and no stats
    player_names = []
    for x in player_totals:
        # Splits name into first and last name. If there is a middle name
        # it is ignored
        name = [x['name'].split()[0], x['name'].split()[-1]]
        # Check if names are already in the list. When players change
        # teams in the middle of the season, Basketball Reference
        # will have separate entries for that player's tenure on each
        # team. We don't want to over represent a name because of this.
        if name not in player_names:
            player_names.append(name)
    
    # Separate the first and last names into their own datasets
    # and lower case them for consistency
    first_names = [x[0].lower() for x in player_names]
    last_names = [x[1].lower() for x in player_names]

    return first_names, last_names

def make_vocab(names):
    """Creates a list of unique characters seen in the given names
        
    Args:
        names (list): A list of strings of names to create vocab
            
    Returns:
        list: Unique characters from names
    
    """
    
    # Initialize list with our designated start and end tokens
    vocab = ['<','>']
    
    # Collect unique characeters
    for name in names:
        for char in list(name):
            if char not in vocab:
                vocab.append(char)
    return vocab

def new_freq_matrix(size):
    """Creates an empty numpy array used for holding frequency
        statistics between each character
        
    Args:
        size (int): Size of matrix, i.e. number of unique characters
            
    Returns:
        ndarray: Zero initialized array of shape (size, size)
    
    """
    freq_matrix = np.zeros((size, size))
    return freq_matrix

def transition_probabilities(names, vocab):
    """Calculates and stores the empirical transition probabilites 
        for every character at every time step in the given dataset
        
    Args:
        names (list): A list of strings of names to compute
            probabilities from
        vocab (list): A list of unique characters in names
            
    Returns:
        dictionary: Transition probabilities array at each time step
    
    """
    
    # Specify our start and end tokens
    start = '<'
    end = '>'
    
    # Get size of the vocab
    size = len(vocab)
    transitions = {}
    
    for name in names:
        # Append the start and end tokens to the name string in order
        # to easily include them in calculations
        name = start+name+end
        
        for i, char in enumerate(name):
            # We iterate until the end token, since no transition can
            # be made at the end token
            if i+1 < len(name):
                freq_matrix = transitions.get(str(i), new_freq_matrix(size))
                state = vocab.index(char)
                next_state = vocab.index(name[i+1])
                freq_matrix[state, next_state] += 1
                transitions[str(i)] = freq_matrix
                
    for i in transitions.keys():
        freq_matrix = transitions[i]
        freq_matrix = np.nan_to_num(freq_matrix/freq_matrix.sum(axis=1)[:,None])
        transitions[i] = freq_matrix
                
    return transitions

def generate_name(transitions, vocab, random=True):
    """Generate a name give then transition probabilities and
        unique characters
        
    Args:
        transitions (dict): Transition probabilities array at each time
            step
        vocab (list): A list of unique characters in names
        random (bool, optional): If True then each character will be
            randomly sampled based on the previous character. If False
            then the name generation will be deterministic by choosing the
            most frequent letter at each time step based on the last letter
            
    Returns:
        str: The generated name
    
    """
    
    # Initialize name with the designated start token
    name = '<'
    current_char = '<'
    
    for i in transitions.keys():
        # Get the transition probabilities for the current time step
        prob_matrix = transitions[str(i)]
        
        # Get the position of the current character
        state = vocab.index(current_char)
        
        # Get the row of transition probs for the current character
        prob_row = prob_matrix[state]
        
        if random:
            # Randomly sample from P(x|c) to choose the next character
            new_state = np.random.choice(len(vocab), p=prob_row)
        else:
            # Or choose the character with highest frequency
            new_state = np.argmax(prob_row)
        
        # Get the new character string based on the position
        new_char = vocab[new_state]
        # Append to the name
        name += new_char
        # Reassign current character for next iteration
        current_char = new_char
        # Break once the end token is chosen
        if current_char == '>':
            break
        
    # Strip the name of the start and end tokens
    return name.strip('<>').capitalize()

def letter_error_rate(fake_name, real_name):
    d = edit_distance(fake_name, real_name)
    result = float(d[len(fake_name)][len(real_name)]) / len(fake_name) * 100
    
    return result
    
def edit_distance(fake_name, real_name):
    d = np.zeros((len(fake_name)+1, len(real_name)+1), dtype=np.uint8)
    for i in range(len(fake_name)+1):
        for j in range(len(real_name)+1):
            if i == 0: 
                d[0][j] = j
            elif j == 0: 
                d[i][0] = i
    for i in range(1, len(fake_name)+1):
        for j in range(1, len(real_name)+1):
            if fake_name[i-1] == real_name[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    return d

def find_most_similar_name(fake_name, names):
    lowest_ler = 100.0
    closest_name = ''
    for name in names:
        ler = letter_error_rate(fake_name.lower(), name.lower())
        if ler < lowest_ler:
            lowest_ler = ler
            closest_name = name
    return closest_name.capitalize()

def main():
    names_first, names_last = get_player_names(list(range(2010,2020)))
    vocab = make_vocab(names_first + names_last)
    transitions_first = transition_probabilities(names_first, vocab)
    transitions_last = transition_probabilities(names_last, vocab)
    
    first = generate_name(transitions_first, vocab)
    last = generate_name(transitions_last, vocab)
    
#    closest_first = find_most_similar_name(first, names_first)
#    closest_last = find_most_similar_name(last, names_last)
    
    print(first, last)
#    print('Closest real names:', closest_first, closest_last)
    
if __name__ == '__main__':
    main()