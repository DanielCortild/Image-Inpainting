We consider the following formulation of the in-painting problem:
$$\min_{Z\in \mathcal H}\frac12\|\mathcal AZ-Z_{\text{corrupt}}\|^2+w\|Z_{(1)}\|_*+w\|Z_{(2)}\|_*$$
Where $\mathcal H=\mathbb R^{N\times M\times 3}$, $\mathcal A$ is the operator that selectes the set of correct entries of $Z$, $Z_{(1)}$ is the matrix $[Z(:,:,0)~Z(:,:,1)~Z(:,:,2)]$, $Z_{(2)}$ is the matrix $[Z(:,:,0)^T~Z(:,:,1)^T~Z(:,:,2)^T]^T$, $\|\cdot \|_*$ denotes the matrix nuclear norm and $w$ is a penality parameter.

If we set $f(Z)=\|Z_{(1)}\|_*$, $g(Z)=\|Z_{(2)}\|_*$, $h(Z)=\frac12\|Z-Z_{\text{corrupt}}\|^2_2$ and $L=\mathcal A$, the inpainting problem fits the context of (40) of the considered paper, namely 
$$\min f(x)+g(x)+h(Lx)$$
with $f,g,h$ closed and convex, $h$ smooth, and $L$ a bounded linear map.