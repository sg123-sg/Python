# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:11:36 2021

@author: Sourav
"""

import numpy as np
#from flask import Flask, render_template,request
import pickle#Initialize the flask App
from flask import render_template,Flask

import flask

app = flask.Flask(__name__, template_folder='templates')
@app.route('/')
def main():
    return render_template('execute.html')
if __name__ == '__main__':
    app.debug = True
    app.run()    