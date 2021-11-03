#lang racket

#|
title : matrix.rkt
explanation : matrix computation with some simple methods.
author : Hiroyuki Shiono
date : 11.02.2021
|#

(provide calc
         calc_a_func
         calc_lst
         calc_num
         calc.sum
         dot
         each
         shape
         transpose)

(define calc
  (lambda (m proc n)
    (map (lambda (subm subn)
           (map proc subm subn))
         m n)))

(define calc_a_func
  (lambda (a_func m)
    (map (lambda (row)
           (map a_func row))
         m)))

(define calc_lst
  (lambda (m proc n)
    (map (lambda (subm num)
           (map (lambda (mval)
                  (proc mval (car num)))
                subm))
         m n)))

(define calc_num
  (lambda (num proc m)
    (map (lambda (subm)
           (map (lambda (val)
                  (proc val num))
                subm))
         m)))

(define calc.sum
  (lambda (m)
    (let ([size (car (shape m))])
      (let loop ([i 0]
                 [output '()])
        (if (equal? i size)
            output
            (loop (+ i 1) (append output (list (list (apply + (jth m i)))))))))))

(define dot
  (lambda (x y)
    (let ([size (car (shape x))])
      (let loop ([i 0]
                 [output '()])
        (if (= i size)
            output
            (loop (+ i 1) (append output (list
                                          (map (lambda (suby)
                                                 (apply +
                                                        (map (lambda (num sy)
                                                               (* num sy))
                                                             (jth x i)
                                                             suby)))
                                               (transpose y))))))))))


(define shape
  (lambda (m)
    (cons (length m) (length (car m)))))


(define ith
  (lambda (m i)
    (map (lambda (lst)
           (list-ref lst i))
         m)))

(define jth
  (lambda (m i)
    (list-ref m i)))

(define transpose
  (lambda (m)
    (let ([size (cdr (shape m))])
      (let loop ([i 0]
                 [output '()])
        (if (= i size)
            output
            (loop (+ i 1) (append output (list (ith m i)))))))))

(define each
  (lambda (x i)
    (map (lambda (lst)
           (list (list-ref lst i)))
         x)))



