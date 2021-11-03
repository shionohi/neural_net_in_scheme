#lang racket

(require "activation_function.rkt")
(require "matrix.rkt")

#|
title : neural_net.rkt
explanation : a simple implementation of neural net with one hidden layer
author : Hiroyuki Shiono
date : 11.02.2021
|#

;;; proc : **
;;; parameter : x, a number
;;; output : returns (* x x)
(define xx
  (lambda (x)
    (* x x)))

;;; proc : layer.init
;;; purpose : create the layer where all values in the layer are randomly initialized
;;; parameter : prev, an integer, the number of inputs
;;;             neurons, an integer, the number of neurons
;;; precondition : both prev and neurons have to be integers
;;; postcondition : returns prev * neurons matrix where all values are random double vals between 0 and 1

(define layer.init
  (lambda (prev neurons)
    (let loop ([i 0]
               [output '()])
      (if (= prev i)
          output
          (loop (+ i 1) (append output (list (layer.init.helper neurons))))))))

(define layer.init.helper
  (lambda (neurons)
    (let loop ([i 0]
               [output '()])
      (let ([r (* (random 1000) 0.001)])
        (if (= neurons i)
            output
            (loop (+ i 1) (append output (list r))))))))

;;; proc : initialize_parameters
;;; purpose : initialize w1, b1, w2, and b2
;;; parameter : variables, an integer, the number of variables in x
;;;             neurons, an integer, the numebr of neurons in the hidden layer
;;;             output_size, an integer
;;; output : a list of matrices, (list w1 b1 w2 b2)

(define initialize_parameters
  (lambda (variables neurons output_size)
    (let ([w1 (layer.init neurons variables)]
          [b1 (layer.init neurons 1)]
          [w2 (layer.init output_size neurons)]
          [b2 (layer.init output_size 1)])
      (list w1 b1 w2 b2))))

;;; proc : forward_propagation
;;; purpose : compute output of x with given parameters
;;; parameter : x, a matrix
;;;             parameters, a list of matrices
;;; output : a list of matrices, (list z1 a1 z2 a2)

(define forward_propagation
  (lambda (x parameters)
    (let ([w1 (list-ref parameters 0)]
          [b1 (list-ref parameters 1)]
          [w2 (list-ref parameters 2)]
          [b2 (list-ref parameters 3)])
      (let ([z1 (calc_lst (dot w1 x) + b1)])
        (let ([a1 (calc_a_func tanh z1)])
          (let ([z2 (calc_lst (dot w2 a1) + b2)])
            (let ([a2 (calc_a_func sigmoid z2)])
              (list a1 a2 z1 z2))))))))

;;; proc : cost_function
;;; purpose : compute the error of a2 and y
;;; parameter : both a2 and y have to be matrixes with numbers
;;; postcondition : returns a number

(define cost_function
  (lambda (a2 y)
    (let ([m (cdr (shape a2))])
      (let loop ([i 0]
                 [temp 0])
        (if (= i m)
            (/ m temp)
            (loop (+ i 1)
                  (let ([a (car (list-ref a2 i))]
                        [y (car (list-ref y i))])
                    (+ temp (loss a y)))))))))

;;; proc : loss
;;; purpose : compute the loss of a and y using binary cross entropy
;;; parameter : both a and y have to be numbers
;;; postcondition : returns a number
(define loss
  (lambda (a y)
    (- (* y (log a)) (* (- 1 y) (log (- 1 a))))))

;;; proc : backward_propagation
;;; purpose : compute the derivatives withgiven parameters
;;; parameter : params, a list of matrices
;;;             params2, a list of matrices
;;;             x, a matrix
;;;             y, a matrix
;;;             parameters, a list of matrices
;;; output : a list of matrices, (list dw1 dw2 db2)

(define backward_propagation
  (lambda (params params2 x y)
    (let ([size (cdr (shape x))]
          [w1 (list-ref params 0)]
          [b1 (list-ref params 1)]
          [w2 (list-ref params 2)]
          [b2 (list-ref params 3)]
          [a1 (list-ref params2 0)]
          [a2 (list-ref params2 1)]
          [z1 (list-ref params2 2)]
          [z2 (list-ref params2 3)])
      (let ([dz2 (calc_lst a2 - y)]) 
        (let ([dw2 (calc_num (/ 1 size) * (dot dz2 (transpose a1)))])
          (let ([db2 (calc_num (/ 1 size) * (calc.sum dz2))])
            (let ([dz1 (calc (dot (transpose w2) dz2) * (calc_num 1 - (calc_a_func xx a1)))])
              (let ([dw1 (calc_num (/ 1 size) * (dot dz1 (transpose x)))])
                (let ([db1 (calc_num (/ 1 size) * (calc.sum dz1))])
                  (list dw1 db1 dw2 db2))))))))))

;;; proc : update_params
;;; purpose : compute the derivatives withgiven parameters
;;; parameter : params, a list of matrices
;;;             grads, a list of matrices
;;;             alpha, a number, usually 0 <= alpha < 1
;;;             parameters, a list of matrices
;;; output : a list of matrices, (list new_w1 new_b1 new_w2 new_b2)

(define update_params
  (lambda (params grads alpha)
    (let ([w1 (list-ref params 0)]
          [b1 (list-ref params 1)]
          [w2 (list-ref params 2)]
          [b2 (list-ref params 3)]
          [dw1 (list-ref grads 0)]
          [db1 (list-ref grads 1)]
          [dw2 (list-ref grads 2)]
          [db2 (list-ref grads 3)])
      (let ([new_w1 (calc w1 - (calc_num alpha * dw1))]
            [new_b1 (calc b1 - (calc_num alpha * db1))]
            [new_w2 (calc w2 - (calc_num alpha * dw2))]
            [new_b2 (calc b2 - (calc_num alpha * db2))])
        (list new_w1 new_b1 new_w2 new_b2)))))

;;; proc : nn
;;; purpose : compute the coefficient using neural network
;;; parameter : x, a matrix
;;;             y, a matrix
;;;             alpha, a number, usually 0 <= alpha < 1
;;;             iterations, an integer
;;; output : a list of coefficient matrices, (list w1 b1 w2 b2)

(define nn
  (lambda (x y neurons alpha iterations)
    (let ([params (initialize_parameters (car (shape x)) neurons 1)])
      (let loop ([i 0]
                 [w1 (list-ref params 0)]
                 [b1 (list-ref params 1)]
                 [w2 (list-ref params 2)]
                 [b2 (list-ref params 3)])
        (if (= i iterations)
            (list w1 b1 w2 b2)
            (let ([params2 (forward_propagation x (list w1 b1 w2 b2))])
              (let ([a2 (list-ref params2 3)])
                (let ([grads (backward_propagation (list w1 b1 w2 b2) params2 x y)])
                  (let ([updated (update_params (list w1 b1 w2 b2) grads alpha)])
                    (loop (+ i 1)
                          (list-ref updated 0)
                          (list-ref updated 1)
                          (list-ref updated 2)
                          (list-ref updated 3)))))))))))

;;; proc : predict
;;; purpose : compute the output using forward propagation
;;; parameter : x, a matrix
;;;             params, a list of w1 b1 w2 b2
;;; output : 1 if true, 0 if false.

(define predict
  (lambda (x params)
    (let ([val (list-ref (forward_propagation x params) 3)])
      (if (> (car (car val)) 0.5)
          1
          0))))


;;; sample input
;;; x represents the inputs which size is 2 with 2 variables
(define x (list (list 1 2)
                (list 3 4)))
;;; y represents the inputs which size is 2
(define y (list (list 1 0)))
;;; returns to a list of coefficient
(nn x y 4 0.05 2000)

                      
        
      
        
