#include <miopen/miopen.h>
#include <miopen/tensor_view_utils.hpp>

#include <cmath>

template <typename Tgpu, typename Tref>
int mloSigmoidFocalLossFwdRunHost(Tgpu* input,
                                  miopenTensorDescriptor_t inputDesc,
                                  Tgpu* target,
                                  miopenTensorDescriptor_t targetDesc,
                                  Tref* outputHost,
                                  miopenTensorDescriptor_t outputDesc,
                                  float alpha,
                                  float gamma,
                                  miopenLossReductionMode_t reduction)
{
    auto input_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto target_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto output_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(outputDesc));

    size_t inputSize = miopen::deref(inputDesc).GetElementSize();
    double loss_sum  = 0;

    for(size_t id = 0; id < inputSize; ++id)
    {
        tensor_layout_t<5> idx(input_tv, id);

        auto i = input[input_tv.get_tensor_view_idx(idx)];
        auto t = target[target_tv.get_tensor_view_idx(idx)];

        float sig    = 1 / (1 + std::exp(-i));
        float ceLoss = -(t * std::log(sig) + (1 - t) * std::log(1 - sig));
        float sigT   = sig * t + (1 - sig) * (1 - t);
        float loss   = ceLoss * std::pow(1 - sigT, gamma);

        if(alpha >= 0)
        {
            float alphaT = alpha * t + (1 - alpha) * (1 - t);
            loss         = alphaT * loss;
        }

        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
        {
            outputHost[output_tv.get_tensor_view_idx(idx)] = static_cast<Tref>(loss);
        }
        else
        {
            loss_sum += loss;
        }
    }
    if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
        loss_sum /= inputSize;
    if(reduction != MIOPEN_LOSS_REDUCTION_NONE)
        outputHost[0] = static_cast<Tref>(loss_sum);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int mloSigmoidFocalLossBwdRunHost(Tgpu* input,
                                  miopenTensorDescriptor_t inputDesc,
                                  Tgpu* target,
                                  miopenTensorDescriptor_t targetDesc,
                                  Tgpu* doutput,
                                  miopenTensorDescriptor_t doutputDesc,
                                  Tref* dinput,
                                  miopenTensorDescriptor_t dinputDesc,
                                  Tref* dtarget,
                                  miopenTensorDescriptor_t dtargetDesc,
                                  float alpha,
                                  float gamma,
                                  miopenLossReductionMode_t reduction)
{
    auto input_tv   = miopen::get_inner_expanded_tv<5>(miopen::deref(inputDesc));
    auto target_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(targetDesc));
    auto doutput_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(doutputDesc));
    auto dinput_tv  = miopen::get_inner_expanded_tv<5>(miopen::deref(dinputDesc));
    auto dtarget_tv = miopen::get_inner_expanded_tv<5>(miopen::deref(dtargetDesc));

    size_t inputSize = miopen::deref(inputDesc).GetElementSize();

    for(size_t id = 0; id < inputSize; ++id)
    {
        tensor_layout_t<5> idx(input_tv, id);

        auto i  = input[input_tv.get_tensor_view_idx(idx)];
        auto t  = target[target_tv.get_tensor_view_idx(idx)];
        auto dO = doutput[0];
        if(reduction == MIOPEN_LOSS_REDUCTION_NONE)
            dO = doutput[doutput_tv.get_tensor_view_idx(idx)];

        float p       = 1 / (1 + std::exp(-i));
        float ceLoss  = -(t * std::log(p) + (1 - t) * std::log(1 - p));
        float pT      = p * t + (1 - p) * (1 - t);
        float powPt   = std::pow(1 - pT, gamma);
        float alpha_t = alpha * t + (1 - alpha) * (1 - t);

        if(dinput)
        {
            float dpdi      = std::exp(-i) / std::pow(1 + std::exp(-i), 2);
            float dcelossdi = (-t / p + (1 - t) / (1 - p)) * dpdi;
            float dpowptdi  = gamma * std::pow(1 - pT, gamma - 1) * (1 - 2 * t) * dpdi;

            // L = ce_loss * pow_pt => dL/di = dceloss/di * pow_pt + ce_loss * dpowpt/di
            float dLdi = dcelossdi * powPt + ceLoss * dpowptdi;
            float grad = dO * dLdi;

            if(alpha >= 0)
                grad *= alpha_t;
            if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
                grad /= inputSize;
            dinput[dinput_tv.get_tensor_view_idx(idx)] = static_cast<Tref>(grad);
        }

        if(dtarget)
        {
            auto dcelossdt = -std::log(p) + std::log(1 - p);
            auto dpowptdt  = gamma * std::pow(1 - pT, gamma - 1) * (1 - 2 * p);
            // L = ce_loss * pow_pt => dL/dt = dceloss/dt * pow_pt + ce_loss * dpowpt/dt
            auto dLdt       = dcelossdt * powPt + ceLoss * dpowptdt;
            auto gradTarget = dO * dLdt;

            if(alpha >= 0)
            {
                // alpha_t * dL/dt + dalpha_t/dt * L
                gradTarget = dO * (alpha_t * dLdt + (2 * alpha - 1) * ceLoss * powPt);
            }
            if(reduction == MIOPEN_LOSS_REDUCTION_MEAN)
                gradTarget /= inputSize;
            dtarget[dtarget_tv.get_tensor_view_idx(idx)] = static_cast<Tref>(gradTarget);
        }
    }

    return miopenStatusSuccess;
}
