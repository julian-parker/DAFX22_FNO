# mat2np
Saves MATLAB array into a pickled Numpy array. 
(Currently only tested in Python 3) 

Example usage :

in MATLAB :   

     a = [ 1.2, 3.5, 4.3 ]; 
     mat2np(a, 'a.pkl', 'float64') 
   
then in Python :   

    import pickle
    with open('a.pkl', 'rb') as fin :
        a = pickle.load(fin) 
        
    
