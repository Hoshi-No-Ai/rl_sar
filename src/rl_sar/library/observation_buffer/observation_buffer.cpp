/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "observation_buffer.hpp"

ObservationBuffer::ObservationBuffer() {}

ObservationBuffer::ObservationBuffer(int num_envs,
                                     int num_obs,
                                     int include_history_steps)
    : num_envs(num_envs),
      num_obs(num_obs),
      include_history_steps(include_history_steps)
{
    num_obs_total = num_obs * include_history_steps;
    obs_buf = torch::zeros({num_envs, num_obs_total}, torch::dtype(torch::kFloat32));
}

void ObservationBuffer::reset(std::vector<int> reset_idxs, torch::Tensor new_obs)
{
    // TODO: 需要在刚开始时reset，来确保网络一开始的输入正确
    std::vector<torch::indexing::TensorIndex> indices;
    for (int idx : reset_idxs)
    {
        indices.push_back(torch::indexing::Slice(idx));
    }
    obs_buf.index_put_(indices, new_obs.repeat({1, include_history_steps}));
}

void ObservationBuffer::insert(torch::Tensor new_obs)
{
    // new_obs会插入到ObservationBuffer的末尾
    // Shift observations back.
    torch::Tensor shifted_obs = obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(num_obs, num_obs * include_history_steps)}).clone();
    obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(0, num_obs * (include_history_steps - 1))}) = shifted_obs;

    // Add new observation.
    obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(-num_obs, torch::indexing::None)}) = new_obs;
}

torch::Tensor ObservationBuffer::get_obs_vec(std::vector<int> obs_ids)
{
    // 假如observations_history是[5, 4, 3, 2, 1, 0]，那么会从ObservationBuffer的开始读取到末尾并pushback到obs中
    // 这样new_obs会位于obs的开头
    std::vector<torch::Tensor> obs;
    for (int i = obs_ids.size() - 1; i >= 0; --i)
    {
        int obs_id = obs_ids[i];
        int slice_idx = include_history_steps - obs_id - 1;
        obs.push_back(obs_buf.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(slice_idx * num_obs, (slice_idx + 1) * num_obs)}));
    }
    return torch::cat(obs, -1);
}
