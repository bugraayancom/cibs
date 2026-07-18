use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;

use crate::error::{Error, Result};

/// Checkpoint tensors loaded from safetensors file(s), keyed by HF
/// state-dict name (e.g. `thinker.model.layers.0.self_attn.q_proj.weight`).
pub struct Weights {
    map: HashMap<String, Array>,
}

impl Weights {
    /// Load `model.safetensors` (or all shards listed in
    /// `model.safetensors.index.json`) from a model directory.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let single = model_dir.join("model.safetensors");
        let index = model_dir.join("model.safetensors.index.json");

        let mut map = HashMap::new();
        if single.is_file() {
            map.extend(load_file(&single)?);
        } else if index.is_file() {
            let text = std::fs::read_to_string(&index).map_err(|e| Error::io(&index, e))?;
            let parsed: serde_json::Value = serde_json::from_str(&text)?;
            let weight_map = parsed
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| Error::Config("index.json missing weight_map".into()))?;
            let mut shards: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            shards.sort();
            shards.dedup();
            for shard in shards {
                map.extend(load_file(&model_dir.join(shard))?);
            }
        } else {
            return Err(Error::Config(format!(
                "no model.safetensors or model.safetensors.index.json in {}",
                model_dir.display()
            )));
        }

        Ok(Weights { map })
    }

    /// Remove and return a tensor by name.
    pub fn take(&mut self, name: &str) -> Result<Array> {
        self.map
            .remove(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    }

    /// Whether the checkpoint (still) contains a tensor with this name.
    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

fn load_file(path: &Path) -> Result<HashMap<String, Array>> {
    Array::load_safetensors(path)
        .map_err(|e| Error::Backend(format!("failed to load {}: {e}", path.display())))
}
