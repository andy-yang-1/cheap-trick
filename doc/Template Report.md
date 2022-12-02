## Gemm Template Report



#### Outline

- Implement
- Regular Shape Performance
- Irregular Shape Performance
- Problems



#### Implement

- General Structure

  In general, there is not much difference between template and manual script. The entire template uses a **2-stage pipeline with double buffer**, and the **local memory stage** computation block is manually added to the template. The entire access sequence follows **Global -> local -> shared -> local -> computation**.

- Details

  - bank conflicts

    The manual script uses quadrature to avoid bank conflict, however, this access method has relatively strict requirements on shape. Therefore, my current strategy is to use padding when the padding memory is less than 5%. However, I have to abandon this method when it is larger than 5%.

  - vectorize

    Vectorize and bank conflict have a relatively similar situation. After many experiments, we found that the vectorize of float4 works best. Therefore, in most cases I choose to use a padding of 4, which will give a big speedup when accessing the memory.



#### Regular Shape Performance

|        shape         |    cutlass     |     tvmGen     |
| :------------------: | :------------: | :------------: |
|  M=512 N=512 K=512   | 4214.23 GFLOPS | 8283.59 GFLOPS |
| M=1024 N=1024 K=1024 | 17250.7 GFLOPS | 17995.6 GFLOPS |
| M=2048 N=2048 K=2048 | 16352.4 GFLOPS | 17472.0 GFLOPS |
| M=3072 N=3072 K=3072 | 16224.3 GFLOPS | 16374.9 GFLOPS |
| M=4096 N=4096 K=4096 | 14899.7 GFLOPS | 16946.7 GFLOPS |
| M=8192 N=8192 K=8192 | 12008.5 GFLOPS | 18311.4 GFLOPS |

_Notes: This test uses the cuda measure, which will be slower than the actual speed. Taking M=N=K=1024 as an example, the actual speed can reach 19278 GFLOPS (given by the tvm test)._

#### Irregular Shape Performance

- with padding

  When the bank conflict is obscured by padding, the overall performance is not much different from the regular shape, and there is an overall performance dip of about 5% due to the extra space computed, but aligning with cutlass can be achieved.

- without padding

  This strategy performs very poorly compared to the padding-enabled part. It is not possible to align cutlass, but only perform slightly better than the existing tvm gemm.



#### Problems

- irregular shape without padding performs bad

- inaccurate speed prediction

  When I use the evolutionary strategy, tvm is very inaccurate in predicting model performance. Using M=N=K=1024 as an example, tvm's prediction is 16934 GFLOPS, but in fact it can perform up to 18771 GFLOPS. I am not quite sure if this difference will affect the model selection. I am afraid that good models may be misjudged and screened out by the filtering.

- robustness

  The template requires at least 3000 rounds to get a well-performed model. For irregular shapes, at least 5000 rounds are needed.