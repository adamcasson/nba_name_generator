#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:19:37 2019

@author: adam
"""

from typing import List, Optional, Union

import numpy as np
from basketball_reference_web_scraper import client


def get_player_names(season_end_years: Union[int, List[int]]=2019):
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
    if isinstance(season_end_years, int):
        season_end_years = [season_end_years]
        
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
    
    # Separate the first and last names into their own lists
    # and lower case them for consistency
    first_names = [x[0].lower() for x in player_names]
    last_names = [x[1].lower() for x in player_names]

    return first_names, last_names


class TimeVaryingMarkovLanguageModel:
    def __init__(self, start_token: str='<', end_token: str='>'):
        self.start_token = start_token
        self.end_token = end_token
        self.samples_fit = None
        self.states = None
        self.transitions = None
    
    def _unique_states(self, samples: List[str]):
        """Creates a list of unique characters seen in the given names
        
        Args:
            samples (list): A list of strings of tokens to create vocab
                
        Returns:
            list: Unique characters from sample tokens
        
        """
        
        # Initialize list with our designated start and end tokens
        states = [self.start_token, self.end_token]
        
        # Collect unique characeters
        for token in samples:
            for char in list(token):
                if char not in states:
                    states.append(char)
        return states
    
    def _new_freq_matrix(self, size: int):
        """Creates an empty numpy array used for holding frequency
            statistics between each character
            
        Args:
            size (int): Size of matrix, i.e. number of unique characters
                
        Returns:
            ndarray: Zero initialized array of shape (size, size)
        
        """
        freq_matrix = np.zeros((size, size))
        return freq_matrix
    
    def fit(self, samples: List[str]):
        """Calculates and stores the empirical transition probabilites 
        for every character at every time step in the given dataset
        
        Args:
            samples (list): A list of strings of tokens to compute
                probabilities from
        """
        
        self.states = self._unique_states(samples)
        
        # Get size of the vocab
        size = len(self.states)
        transitions = {}
        
        for token in samples:
            # Append the start and end tokens to the name string in order
            # to easily include them in calculations
            token = self.start_token + token + self.end_token
            
            for i, char in enumerate(token):
                # We iterate until the end token, since no transition can
                # be made at the end token
                if i+1 < len(token):
                    freq_matrix = transitions.get(i, self._new_freq_matrix(size))
                    state = self.states.index(char)
                    next_state = self.states.index(token[i+1])
                    freq_matrix[state, next_state] += 1
                    transitions[i] = freq_matrix
                    
        for i in transitions.keys():
            freq_matrix = transitions[i] + 1e-8 # epsilon to avoid div by 0
            freq_matrix = freq_matrix/freq_matrix.sum(axis=1)[:,None]
            transitions[i] = freq_matrix
                    
        self.transitions = transitions
        self.samples_fit = samples
    
    def generate(self, random: Optional[bool]=True):
        """Generate a name give then transition probabilities and
        unique characters
        
        Args:
            random (bool, optional): If True then each character will be
                randomly sampled based on the previous character. If False
                then the name generation will be deterministic by choosing the
                most frequent letter at each time step based on the last letter
                
        Returns:
            str: The generated token
        
        """
        
        # Initialize name with the designated start token
        token = self.start_token
        current_char = self.start_token
        
        for i in self.transitions.keys():
            # Get the transition probabilities for the current time step
            prob_matrix = self.transitions[i]
            
            # Get the position of the current character
            state = self.states.index(current_char)
            
            # Get the row of transition probs for the current character
            prob_row = prob_matrix[state]
            
            if random:
                # Randomly sample from P(x|c) to choose the next character
                new_state = np.random.choice(len(self.states), p=prob_row)
            else:
                # Or choose the character with highest frequency
                new_state = np.argmax(prob_row)
            
            # Get the new character string based on the position
            new_char = self.states[new_state]
            # Append to the name
            token += new_char
            # Reassign current character for next iteration
            current_char = new_char
            # Break once the end token is chosen
            if current_char == self.end_token:
                break
            
        # Strip the name of the start and end tokens
        return token.strip(self.start_token + self.end_token).capitalize()


def main():
    names_first, names_last = get_player_names(list(range(2019,2020)))
    
    first_model = TimeVaryingMarkovLanguageModel()
    first_model.fit(names_first)
    last_model = TimeVaryingMarkovLanguageModel()
    last_model.fit(names_last)
    
    first = first_model.generate()
    last = last_model.generate()

    print(first, last)

    
if __name__ == '__main__':
    main()