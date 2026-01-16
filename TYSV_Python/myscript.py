# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 14:40:39 2025

@author: gleblanc
"""

def main():
    print("The main function is running!")
    print("The __name__ parameter is:" + __name__)
    
if __name__ == "__main__":
    main()
    
else:
    print("The mail function is not running")
    print("The __name__ parameter is" + __name__)
    