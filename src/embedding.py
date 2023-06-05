import pandas as pd
import csv
import sys

from bio_embeddings.embed import SeqVecEmbedder as SeqVecEmbedder
from bio_embeddings.embed import ESM1bEmbedder as Esm1bEmbedder
from bio_embeddings.embed import ESMEmbedder as EsmEmbedder
from bio_embeddings.embed import ProtTransT5XLU50Embedder as Prott5Embedder
from bio_embeddings.embed import OneHotEncodingEmbedder as OneHotEmbedder

import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"
class Embedding:
    """
    This class is used for generating embeddings of protein sequences using different models. 

    Attributes:
    in_file (str): Path to the input file containing protein sequences.
    out_file (str): Path to the output file where the embeddings will be stored.
    level (str): Level of embedding ('protein', 'residue', or 'both').
    embed (str): The embedding model to use ('prott5', 'esm', 'esm1b', or 'onehot').
    """
    def __init__(self, in_file, out_file, level='residue', embed='prott5'):
        self.in_file = in_file
        self.out_file = out_file
        self.out_dir = f'{DATA_DIR}/fasta'
        self.level = level
        self.embed = embed

    def embedding(self):
        """
        This function generates the embeddings of protein sequences.

        Returns:
        The embeddings of protein sequences as a pandas DataFrame.
        """
        # reading sequence
        seq_dict = self.__read_file(filepath=self.in_file)
        if self.embed == 'onehot':
            vect_len = 21
        elif ((self.embed == 'esm') | (self.embed == 'esm1b')):
            vect_len = 1280
        else:
            vect_len = 1024
        # checking for level
        if self.level=='protein':
            l = 0
        elif self.level=='residue':
            l = 1
        elif self.level=='both':
            l = 2
        # file initialization
        if not(l==0): #create two files for both option
            self._file__initialize(f'./{self.out_dir}/{self.out_file}_residue_{self.embed}.csv',vect_len,l)
        if not(l==1):
            self.__file_initialize(f'./{self.out_dir}/{self.out_file}_protein_{self.embed}.csv',vect_len)
        # Running embeddings/writing to csv row by row
        return self.__embed_one_file(sequences=seq_dict, out_dir=self.out_dir, out_file=self.out_file, embed=self.embed,  is_residue=l)
    
    def __read_file(self, filepath):
        """
        This function reads a FASTA file and returns a dictionary of {protein ID: sequence}.

        Parameters:
        filepath (str): Path to the FASTA file.

        Returns:
        dictionary (dict): A dictionary where the keys are protein IDs and the values are protein sequences.
        """
        dictionary = {}
        with open(filepath) as fasta_file:
            seq = ''
            for line in fasta_file:
                line=line.rstrip()
                if line.startswith(">"):
                    if seq.__len__():
                        dictionary[name] = seq
                    name = line
                    seq = ''
                else:
                    seq = seq+line
            dictionary[name] = seq
            
        dic2=dict(sorted(dictionary.items(),key= lambda x:len(x[1]), reverse=True))
        return dic2

    def __file_initialize(self, target_file, vect_len, is_residue=False):
        """
            This function initializes a CSV file with the appropriate column names.

            Parameters:
            target_file (str): Path to the target CSV file.
            vect_len (int): Vector length (specific to different models).
            is_residue (bool, optional): If True, initializes both protein and residue level files. If False, only initializes protein file. Default is False.

            Returns:
            True if the CSV file is successfully initialized.
        """

        random_list = ['sv'+str(x) for x in range(1, vect_len+1)]
        random_list.append('ProteinID')
        if is_residue:
            random_list.append('ResidueID')
        # writing to csv file 
        with open(target_file, 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
            # writing the fields 
            csvwriter.writerow(random_list)
        return True

    def __write_csv(self, target_file, row):
        """
        This function writes embbedding row to a CSV file.

        Parameters:
        target_file (str): Path to the target CSV file.
        row (list): The row to be written to the CSV file.

        Returns:
        True if the row is successfully written to the CSV file.
        """

        with open(target_file, 'a') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile) 
            # writing the fields 
            csvwriter.writerow(row)
        return True
    
    def __embed_one_file(self, sequences, out_dir, out_file, embed='prott5', is_residue=True):
        """
        This function generates and writes the embeddings of protein sequences to a CSV file.

        Parameters:
        sequences (dict): A dictionary where the keys are protein names and the values are protein sequences.
        out_dir (str): Directory where the CSV file will be saved.
        out_file (str): Name of the CSV file.
        embed (str, optional): The embedding model to use. It can be 'prott5', 'esm', 'esm1b', 'seq', or 'onehot'. Default is 'prott5'.
        is_residue (bool, optional): If True, generates both protein and residue level embeddings. If False, only generates protein level embeddings. Default is True.

        Returns:
        Path to the generated CSV file.
        """
        if embed == 'esm1b':
            embedder = Esm1bEmbedder()
        elif embed == 'esm':
            embedder = EsmEmbedder()
        elif embed == 'seq':
            embedder = SeqVecEmbedder()
        elif embed == 'prott5':
            embedder = Prott5Embedder()
        elif embed == 'onehot':
            embedder = OneHotEmbedder()
        
        for protein in sequences:
            reduced_embeddings = []

            embeddings = embedder.embed(sequences[protein]) # using single embed since seach element is a STRING
            if not(is_residue==1):
                # protein level
                reduced_embeddings = embedder.reduce_per_protein(embeddings)
                reduced_embeddings = reduced_embeddings.tolist()
                reduced_embeddings.append(protein)
                self.__write_csv(f'./{out_dir}/{out_file}_protein_{embed}.csv', reduced_embeddings)
                reduced_embeddings.clear()
        
            if not(is_residue==0):
            # residue level
                r_embeddings = embeddings
                if embed == 'seq':
                    r_embeddings = r_embeddings.sum(0)
                residue_embeddings = r_embeddings.tolist()
                for idx, char in enumerate(sequences[protein]):
                    res_embed = residue_embeddings[idx]
                    res_embed.append(protein)
                    res_embed.append(char+str(idx+1))
                    self.__write_csv(f'./{out_dir}/{out_file}_residue_{embed}.csv', res_embed)
                    # res_embed.clear()
        return f'./{out_dir}/{out_file}_residue_{embed}.csv' if not(is_residue==0) else f'./{out_dir}/{out_file}_protein_{embed}.csv'