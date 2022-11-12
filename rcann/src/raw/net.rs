use std::fmt::{Debug, Formatter, Write};
use rand::Rng;
use rand_distr::Normal;
use crate::activation::ActivationFn;
use crate::raw::util::{mat_mul_b_transpose, resize_if_needed};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub prev_size: usize,
    pub size: usize,
    #[serde(with = "serialize_vec")]
    pub weights: Box<[f32]>,
    #[serde(with = "serialize_vec")]
    pub biases: Box<[f32]>,
    #[serde(skip)]
    pub activations: Vec<f32>,
    #[serde(skip)]
    pub outputs: Vec<f32>,
    pub activation_fn: ActivationFn,
}

impl Layer {

    pub fn new(
        prev_size: usize,
        size: usize,
        activation_fn: ActivationFn,
        rng: &mut rand::rngs::StdRng,
    ) -> Self {
        let std = (2.0 / (prev_size + size) as f32).sqrt();
        let dist = Normal::new(0.0, std).unwrap();
        return Layer {
            prev_size,
            size,
            activations: Vec::new(),
            outputs: Vec::new(),
            weights: rng.sample_iter(dist).take(size * prev_size).collect(),
            biases: vec![0.0; size].into_boxed_slice(),
            activation_fn,
        }
    }

    pub fn forward(&mut self, batch_size: usize, input: &[f32]) -> &[f32] {
        debug_assert_eq!(input.len(), batch_size * self.prev_size);
        let buff_len = batch_size * self.size;
        resize_if_needed(&mut self.activations, buff_len);
        resize_if_needed(&mut self.outputs, buff_len);
        mat_mul_b_transpose(
            batch_size,
            self.prev_size,
            self.size,
            1.0,
            0.0,
            input,
            &self.weights,
            &mut self.activations
        );
        self.activation_fn.compute_slice(self.size,&mut self.outputs, &self.activations);
        &self.outputs
    }

}

#[derive(Clone, Serialize, Deserialize)]
pub struct Net {
    pub first: Layer,
    pub hidden: Box<[Layer]>,
    pub last: Layer,
}

impl Net {

    pub fn new(first: Layer, hidden: Vec<Layer>, last: Layer) -> Self {
        let mut prev = &first;
        for cur in &hidden {
            assert_eq!(cur.prev_size, prev.size);
            prev = cur;
        }
        assert_eq!(last.prev_size, prev.size);
        return Net {
            first,
            hidden: hidden.into_boxed_slice(),
            last,
        }
    }

    pub fn forward(&mut self, batch_size: usize, input: &[f32]) -> &[f32] {

        assert_eq!(input.len(), batch_size * self.input_len());

        // compute first layer from input
        let mut prev_outputs = self.first.forward(batch_size, input);

        // pass hidden layers
        for layer in self.hidden.iter_mut() {
            prev_outputs = layer.forward(batch_size, prev_outputs)
        }

        // compute last layer
        self.last.forward(batch_size, prev_outputs)
    }

    #[inline]
    pub fn input_len(&self) -> usize {
        self.first.prev_size
    }

    #[inline]
    pub fn output_len(&self) -> usize {
        self.last.size
    }

}

fn fmt_slice_elements(f: &mut Formatter<'_>, slice: &[f32]) -> std::fmt::Result  {
    let mut first = true;
    for &x in slice {
        if first {
            first = false;
        } else {
            f.write_str(", ")?;
        }
        f.write_str(x.to_string().as_str())?;
    }
    Ok(())
}

fn fmt_slice(f: &mut Formatter<'_>, slice: &[f32]) -> std::fmt::Result {
    f.write_char('[')?;
    if slice.len() > 10 {
        fmt_slice_elements(f, &slice[..5])?;
        f.write_str(" ... ")?;
        fmt_slice_elements(f, &slice[slice.len()-5..])?;
    } else {
        fmt_slice_elements(f, slice)?;
    }
    f.write_char(']')
}

impl Debug for Layer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer {{ size: {}, activation_fn: {:?}, weights: ", self.size, self.activation_fn)?;
        fmt_slice(f, &self.weights)?;
        f.write_str(", biases: ")?;
        fmt_slice(f, &self.biases)?;
        f.write_str(" }")
    }
}

impl Debug for Net {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Net {\n")?;
        f.write_str("   first: ")?;
        self.first.fmt(f)?;
        f.write_str(",\n   hidden: [\n")?;
        for layer in self.hidden.iter() {
            f.write_str("      ")?;
            layer.fmt(f)?;
        }
        f.write_str("\n   ],\n")?;
        f.write_str("   last: ")?;
        self.last.fmt(f)?;
        f.write_str("\n}")
    }
}


mod serialize_vec {
    use std::fmt::Formatter;
    use serde::{Deserializer, Serializer};
    use serde::de::{Error, Visitor};

    struct Vec32Visitor;
    
    impl<'de> Visitor<'de> for Vec32Visitor {
        type Value = Vec<f32>;
        fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
            formatter.write_str("a base64-encoded string encoding a f32 array")
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> where E: Error {
            match base64::decode(v) {
                Ok(data) => {
                    if data.len() % 4 != 0 {
                        Err(E::custom("byte length not a multiple of 4"))
                    } else {
                        let mut res = Vec::with_capacity(data.len() / 4);
                        for chunk in data.chunks_exact(4) {
                            let f32bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                            res.push(f32::from_be_bytes(f32bytes));
                        }
                        Ok(res)
                    }
                }
                Err(e) => {
                    Err(E::custom(e.to_string()))
                }
            }
        }
    }

    pub fn deserialize<'de, T, D>(deserializer: D) -> Result<T, D::Error>
        where D: Deserializer<'de>, T: From<Vec<f32>>
    {
        deserializer.deserialize_str(Vec32Visitor).map(|vec| vec.into())
    }

    pub fn serialize<T, S>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer, T: AsRef<[f32]>
    {
        let arr = value.as_ref();
        let mut bytes = Vec::with_capacity(arr.len() * 4);
        for &f in arr {
            let [b1, b2, b3, b4] = f.to_be_bytes();
            bytes.push(b1);
            bytes.push(b2);
            bytes.push(b3);
            bytes.push(b4);
        }
        let b64str = base64::encode(bytes);
        serializer.serialize_str(b64str.as_str())
    }

}