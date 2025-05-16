use crate::{
    bgg::{
        circuit::{build_composite_circuit_from_public_and_fhe_dec, Evaluable, PolyCircuit},
        sampler::*,
        BggPublicKey,
    },
    poly::{element::PolyElem, sampler::*, Poly, PolyMatrix, PolyParams},
};
use std::marker::PhantomData;

use super::params::ObfuscationParams;

const TAG_A_RLWE_BAR: &[u8] = b"A_RLWE_BAR";
const TAG_A_PRF: &[u8] = b"A_PRF:";
pub const TAG_BGG_PUBKEY_INPUT_PREFIX: &[u8] = b"BGG_PUBKEY_INPUT:";

pub fn sample_public_key_by_id<K: AsRef<[u8]>, S>(
    sampler: &BGGPublicKeySampler<K, S>,
    params: &<<<S as PolyHashSampler<K>>::M as PolyMatrix>::P as Poly>::Params,
    id: usize,
    reveal_plaintexts: &[bool],
) -> Vec<BggPublicKey<<S as PolyHashSampler<K>>::M>>
where
    S: PolyHashSampler<K>,
{
    sampler.sample(
        params,
        &[TAG_BGG_PUBKEY_INPUT_PREFIX, &(id as u64).to_le_bytes()].concat(),
        reveal_plaintexts,
    )
}

#[derive(Debug, Clone)]
pub struct PublicSampledData<S: PolyHashSampler<[u8; 32]>> {
    pub a_rlwe_bar: S::M,
    pub a_prf: S::M,
    pub packed_input_size: usize,
    pub packed_output_size: usize,
    _s: PhantomData<S>,
}

impl<S: PolyHashSampler<[u8; 32]>> PublicSampledData<S> {
    pub fn sample(obf_params: &ObfuscationParams<S::M>, hash_key: [u8; 32]) -> Self {
        let hash_sampler = S::new();
        let params = &obf_params.params;
        let d = obf_params.d;
        let log_base_q = params.modulus_digits();
        let dim = params.ring_dimension() as usize;
        // input bits, poly of the FHE key, it contains 1 for the FHE key (t)
        let packed_input_size = obf_params.input_size.div_ceil(dim) + 1;
        let packed_output_size = obf_params.public_circuit.num_output() / (2 * log_base_q);
        let a_rlwe_bar =
            hash_sampler.sample_hash(params, hash_key, TAG_A_RLWE_BAR, 1, 1, DistType::FinRingDist);

        let a_prf_raw = hash_sampler.sample_hash(
            params,
            hash_key,
            TAG_A_PRF,
            d + 1,
            packed_output_size,
            DistType::FinRingDist,
        );
        let a_prf = a_prf_raw.modulus_switch(&obf_params.switched_modulus);
        Self { a_rlwe_bar, a_prf, packed_input_size, packed_output_size, _s: PhantomData }
    }
}

pub fn build_final_digits_circuit<P: Poly, E: Evaluable>(
    a_decomposed_polys: &[P],
    b_decomposed_polys: &[P],
    public_circuit: PolyCircuit,
) -> PolyCircuit {
    let log_base_q = a_decomposed_polys.len();
    debug_assert_eq!(b_decomposed_polys.len(), log_base_q);
    let packed_eval_input_size = public_circuit.num_input() - (2 * log_base_q);

    // circuit outputs the cipertext ct=(a,b) as a_base_0, b_base_0, a_base_1, b_base_1, ...
    let mut ct_output_circuit = PolyCircuit::new();
    {
        let inputs = ct_output_circuit.input(packed_eval_input_size);
        let circuit_id = ct_output_circuit.register_sub_circuit(public_circuit);
        let mut public_circuit_inputs = vec![];
        for poly in a_decomposed_polys.iter() {
            let digits = poly.coeffs_digits();
            public_circuit_inputs.push(ct_output_circuit.const_digits_poly(&digits));
        }
        for poly in b_decomposed_polys.iter() {
            let digits = poly.coeffs_digits();
            public_circuit_inputs.push(ct_output_circuit.const_digits_poly(&digits));
        }
        public_circuit_inputs.extend(inputs);
        assert_eq!(public_circuit_inputs.len(), 2 * log_base_q + packed_eval_input_size);
        let pc_outputs = ct_output_circuit.call_sub_circuit(circuit_id, &public_circuit_inputs);
        let mut outputs = Vec::with_capacity(pc_outputs.len());
        // n is the number of ciphertexts
        let n = pc_outputs.len() / (2 * log_base_q);
        for ct_idx in 0..n {
            let ct_offset = ct_idx * 2 * log_base_q;
            for bit_idx in 0..log_base_q {
                let a_index = ct_offset + bit_idx;
                let b_index = ct_offset + log_base_q + bit_idx;
                let a_bit = pc_outputs[a_index];
                let b_bit = pc_outputs[b_index];
                outputs.push(a_bit);
                outputs.push(b_bit);
            }
        }
        ct_output_circuit.output(outputs);
    }

    // actual
    let mut circuit = PolyCircuit::new();
    {
        let inputs = circuit.input(packed_eval_input_size + 1); // + 1 is for -t_bar
        let sub_circuit =
            build_composite_circuit_from_public_and_fhe_dec::<E>(ct_output_circuit, log_base_q);
        let circuit_id = circuit.register_sub_circuit(sub_circuit);
        let outputs = circuit.call_sub_circuit(circuit_id, &inputs);
        circuit.output(outputs);
    }
    circuit
}

pub fn build_u_mask_multi<M: PolyMatrix>(
    params: &<<M as PolyMatrix>::P as Poly>::Params,
    packed_input_size: usize, // L
    level_width: usize,       // w
    depth_j: usize,           // j
    combo_b: usize,           // 0 ≤ b < 2^w
) -> M {
    // L' = 1 (t) + L packed-input slots (evaluator's input + const-one slot)
    let l_dash = 1 + packed_input_size;

    // U_{j,0}  ≡  identity
    if combo_b == 0 {
        return M::identity(params, l_dash, None);
    }

    // Start from the identity and zero out the diagonal entry
    // for every bit that is 1 in combo_b.
    let mut u = M::identity(params, l_dash, None);
    let mut zero = M::P::const_zero(params);

    for r in 0..level_width {
        if (combo_b >> r) & 1 == 1 {
            let idx = depth_j * level_width + r;
            assert!(
                idx < packed_input_size * params.ring_dimension() as usize,
                "index out of range"
            );
            let mut cs = zero.coeffs();
            debug_assert!(idx < cs.len(), "index out of bounds");
            cs[idx] = <M::P as Poly>::Elem::one(&params.modulus());
            zero = M::P::from_coeffs(params, &cs);
        }
    }

    u.set_entry(0, 1, zero);

    u
}

pub fn build_poly_vec<M: PolyMatrix>(
    params: &<<M as PolyMatrix>::P as Poly>::Params,
    inputs: &[bool],
    level_width: usize,
    level: usize,
    input_size: usize,
    minus_t_bar: Option<M::P>,
) -> Vec<M::P> {
    let bits_done = level_width * level;
    let dim = params.ring_dimension() as usize;
    let mut polys: Vec<M::P> = vec![<M::P as Poly>::const_one(params)];

    let mut coeffs = inputs[..bits_done]
        .iter()
        .map(|&b| {
            if b {
                <M::P as Poly>::Elem::one(&params.modulus())
            } else {
                <M::P as Poly>::Elem::zero(&params.modulus())
            }
        })
        .collect::<Vec<_>>();

    coeffs.extend(std::iter::repeat_n(
        <M::P as Poly>::Elem::zero(&params.modulus()),
        input_size - bits_done,
    ));

    polys.extend(coeffs.chunks(dim).map(|c| M::P::from_coeffs(params, c)));
    if let Some(minus_t_bar) = minus_t_bar {
        polys.push(minus_t_bar);
    }
    polys
}

#[cfg(test)]
mod test {
    use num_bigint::BigUint;
    use num_traits::One;

    use super::*;
    use crate::{
        bgg::DigitsToInt,
        poly::{
            dcrt::{DCRTPoly, DCRTPolyParams, DCRTPolyUniformSampler},
            enc::rlwe_encrypt,
            sampler::DistType,
        },
    };

    #[test]
    fn test_build_final_step_circuit() {
        // 1. Set up parameters
        let params = DCRTPolyParams::default();
        let log_q = params.modulus_bits();
        let sampler_uniform = DCRTPolyUniformSampler::new();
        let sigma = 3.0;
        let log_base_q = params.modulus_digits();

        // 2. Create a simple public circuit that takes 2*log_q inputs and outputs them directly
        let mut public_circuit = PolyCircuit::new();
        // inputs: BaseDecompose(ct), eval_input
        // outputs: BaseDecompose(ct) AND eval_input
        {
            let inputs = public_circuit.input((2 * log_base_q) + 1);
            let mut outputs = vec![];
            let eval_input = inputs[2 * log_base_q];
            for ct_input in inputs[0..2 * log_base_q].iter() {
                let muled = public_circuit.and_gate(*ct_input, eval_input);
                outputs.push(muled);
            }
            public_circuit.output(outputs);
        }

        // 3. Generate a random hardcoded key
        let hardcoded_key = sampler_uniform.sample_uniform(&params, 1, 1, DistType::BitDist);

        // 4. Generate RLWE ciphertext for the hardcoded key
        let a_rlwe_bar = sampler_uniform.sample_uniform(&params, 1, 1, DistType::BitDist);
        let t_bar_matrix = sampler_uniform.sample_uniform(&params, 1, 1, DistType::BitDist);

        let b = rlwe_encrypt(
            &params,
            &sampler_uniform,
            &t_bar_matrix,
            &a_rlwe_bar,
            &hardcoded_key,
            sigma,
        );

        // 5. Decompose the ciphertext
        let a_decomposed = a_rlwe_bar.entry(0, 0).decompose_base(&params);
        let b_decomposed = b.entry(0, 0).decompose_base(&params);

        // 6. Build the final circuit with DCRTPoly as the Evaluable type
        let final_circuit = build_final_digits_circuit::<DCRTPoly, DCRTPoly>(
            &a_decomposed,
            &b_decomposed,
            public_circuit,
        );

        // 7. Evaluate the circuit
        let one = DCRTPoly::const_one(&params);

        let mut inputs = vec![one.clone()];
        inputs.push(-(t_bar_matrix.entry(0, 0)).clone());

        let circuit_outputs = final_circuit.eval(&params, &one, &inputs);
        assert_eq!(circuit_outputs.len(), log_q);

        // 8. Extract the output bits
        let output_ints = circuit_outputs
            .chunks(log_q)
            .map(|bits| DCRTPoly::digits_to_int(bits, &params))
            .collect::<Vec<_>>();
        assert_eq!(output_ints.len(), 1);
        let output_bits = output_ints
            .iter()
            .flat_map(|output| output.extract_bits_with_threshold(&params))
            .collect::<Vec<_>>();

        // 9. Verify that the output matches the hardcoded key bits
        assert_eq!(output_bits.len(), params.ring_dimension() as usize);
        assert_eq!(output_bits, hardcoded_key.entry(0, 0).to_bool_vec());
    }

    #[test]
    fn test_simulate_norm_final_bits_circuit() {
        // 1. Set up parameters
        let log_n = 13u32;
        let n = 2u32.pow(log_n);
        let crt_depth = 12;
        let crt_bits = 51;
        let base_bits = 20;
        let params = DCRTPolyParams::new(n, crt_depth, crt_bits, base_bits);
        let log_q = params.modulus_bits();
        debug_assert_eq!(crt_bits * crt_depth, log_q);
        let log_base_q = params.modulus_digits();

        // 2. Create a simple public circuit that takes log_base_q inputs and outputs them directly
        let mut public_circuit = PolyCircuit::new();
        // inputs: BaseDecompose(ct), eval_input
        // outputs: BaseDecompose(ct) AND eval_input
        {
            let inputs = public_circuit.input((2 * log_base_q) + 1);
            let mut outputs = vec![];
            let eval_input = inputs[2 * log_base_q];
            for ct_input in inputs[0..2 * log_base_q].iter() {
                let muled = public_circuit.and_gate(*ct_input, eval_input);
                outputs.push(muled);
            }
            public_circuit.output(outputs);
        }

        let a_rlwe_bar = DCRTPoly::const_max(&params);
        let b = DCRTPoly::const_max(&params);

        let a_decomposed_polys = a_rlwe_bar.decompose_base(&params);
        let b_decomposed_polys = b.decompose_base(&params);
        let final_circuit = build_final_digits_circuit::<DCRTPoly, DCRTPoly>(
            &a_decomposed_polys,
            &b_decomposed_polys,
            public_circuit,
        );

        let packed_input_norms = vec![BigUint::one(), params.modulus().as_ref().clone()];
        let norms = final_circuit.simulate_bgg_norm(
            params.ring_dimension(),
            params.base_bits(),
            packed_input_norms,
        );
        let norm_json = serde_json::to_string(&norms).unwrap();
        use std::{fs::File, io::Write};
        let mut file = File::create(format!(
            "final_bits_norm_n_{}_crt_{}_depth_{}_base_{}.json",
            log_n, crt_bits, crt_depth, base_bits
        ))
        .unwrap();
        file.write_all(norm_json.as_bytes()).unwrap();
    }
}
