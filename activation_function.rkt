#lang racket
#|
title : activation_function.rkt
explanation : provide some popular activation functions
author : Hiroyuki Shiono
date : Oct 31 2021
|#

(provide sigmoid
         tanh)

(define sigmoid
  (lambda (x)
    (/ 1 (+ 1 (exp (* -1 x))))))

(define tanh
  (lambda (x)
    (- (/ 2 (+ 1 (exp (* -2 x)))) 1)))