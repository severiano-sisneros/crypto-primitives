//! This module implements a lattice-based collision resistant hash function based on the Short Integer Solution
//! (SIS) problem. For more details about the SIS problem see Chris Peikart's survey: A Decade of
//! Lattice Cryptography, Definition 4.1.

use crate::{crh::CRHScheme, Error};
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{borrow::Borrow, marker::PhantomData, rand::Rng};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A trait that defines an instance of the SIS problem. N and M correspond to the rows and columns
/// of the matrix A, resp. BETA bounds the norm of the input vector; to ensure collision resistance the
/// norm of the input must be less tha BETA. In general, N is the primary security parameter. M
/// must be greater than N*log(q), where q is the size of the field F. And BETA must be greater
/// than sqrt(N*log(q)). See [On the Concrete Hardness of Learning with
/// Errors](https://eprint.iacr.org/2015/046) for more details.
pub trait SisProblem: Clone {
    const N: usize;
    const M: usize;
    const BETA: usize;
}

#[derive(Clone, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct Parameters<F: PrimeField> {
    /// The N x M matrix A.
    pub a: Vec<Vec<F>>,
}

pub struct CRH<F: PrimeField, S: SisProblem> {
    field: PhantomData<F>,
    sis_problem: PhantomData<S>,
}

impl<F: PrimeField, S: SisProblem> CRH<F, S> {
    /// Creates the N x M matrix A with uniformly random colums whose elements are drawn
    /// from the field F.
    pub fn create_a_matrix<R: Rng>(rng: &mut R) -> Vec<Vec<F>> {
        (0..S::N)
            .map(|_| (0..S::M).map(|_| F::rand(rng)).collect())
            .collect()
    }
}

impl<F: PrimeField, S: SisProblem> CRHScheme for CRH<F, S> {
    /// A vector of length M whose elements are from F.
    type Input = Vec<F>;
    /// A vector of length N whose elements are from F.
    type Output = Vec<F>;
    /// The N x M matrix A.
    type Parameters = Parameters<F>;

    /// Generates the NxM matrix A.
    fn setup<R: Rng>(rng: &mut R) -> Result<Self::Parameters, Error> {
        // The matrix a is a uniformly random matrix of size n x m with elements from F
        let a = Self::create_a_matrix::<R>(rng);
        Ok(Parameters { a })
    }

    /// Evaluates the hash function on the input vector. A simple matrix vector multiplication,
    /// A x input.
    fn evaluate<T: Borrow<Self::Input>>(
        parameters: &Self::Parameters,
        input: T,
    ) -> Result<Self::Output, Error> {
        let input = input.borrow();

        // Input check: input length must be equal to n, this is an unrecoverable error
        if input.len() != S::M {
            panic!(
                "Incorrect input length {:?}, expected {:?}",
                input.len(),
                S::M
            );
        }

        // Input check: the collision resistance of the hash function is parameterized by BETA.
        // By asserting that each element of the input is in the range [0, b-1], we ensure that
        // the norm of the input vector is less than BETA, as required for collision resistance.
        for i in 0..S::N {
            if input[i] >= F::from_bigint((S::BETA as u64).into()).unwrap() {
                panic!(
                    "Input element {:?} is out of range [0, {:?})",
                    input[i],
                    S::BETA
                );
            }
        }

        // Hashing is simply matrix vector multiplication a x input
        let res = mat_vec_mul(&parameters.a, input);
        Ok(res)
    }
}

// Generic function for matrix-vector multiplication
fn mat_vec_mul<F: PrimeField>(matrix: &[Vec<F>], vector: &[F]) -> Vec<F> {
    #[cfg(feature = "parallel")]
    {
        // Parallel version
        return matrix
            .par_iter() // Parallelize over rows
            .map(|row| row.iter().zip(vector).map(|(a, b)| *a * *b).sum()) // Field multiplication + sum
            .collect();
    }

    // Sequential version
    matrix
        .iter()
        .map(|row| row.iter().zip(vector).map(|(a, b)| *a * *b).sum())
        .collect()
}

#[cfg(test)]
mod test {
    use crate::crh::sis::{CRHScheme, SisProblem, CRH};
    use ark_bls12_377::Fr;
    use ark_std::{rand::prelude::*, test_rng};

    #[derive(Clone)]
    struct TestHash;

    impl SisProblem for TestHash {
        const N: usize = 128;
        const M: usize = 3000;
        const BETA: usize = 2 << 8;
    }

    const INPUT_LEN: usize = 3000;
    #[test]
    fn test_sis() {
        let rng = &mut test_rng();
        println!("Setting up parameters");
        let parameters = CRH::<Fr, TestHash>::setup(rng).unwrap();
        println!("Done");
        println!("Generating random input");
        // choose random input in the range [0, beta-1]
        let input: Vec<Fr> = (0..INPUT_LEN)
            .map(|_| Fr::from(rng.gen::<u64>() % (TestHash::BETA as u64)))
            .collect();
        println!("Done");
        println!("Evaluating hash function");
        let _ = CRH::<Fr, TestHash>::evaluate(&parameters, input).unwrap();
        println!("Done");
    }
}
