import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm




# Generate a dataset with two Gaussian components
print('gen data...',end='              \r')
mu1, sigma1 = 2, 1
mu2, sigma2 = -1, 1.5
X1 = np.random.normal(mu1, sigma1, size=2000)
X2 = np.random.normal(mu2, sigma2, size=6000)
X = np.concatenate([X1, X2])
print(f'N1~{(mu1,sigma1)}, N2~{(mu2,sigma2)}')

# Plot the density estimation using seaborn
print('plot kde...',end='              \r')
sns.kdeplot(X)
plt.hist(X,bins=40,density=True,color='grey')
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Density Estimation of X')
plt.show()


# Initialize parameters
print('Init distribution...',end='              \r')
mu1_hat, sigma1_hat = np.mean(X1)+3, np.std(X1)*1.5
mu2_hat, sigma2_hat = np.mean(X2)-3, np.std(X2)*2
pi1_hat, pi2_hat = len(X1) / len(X), len(X2) / len(X)

X_axis = np.linspace(-10,10,4000)
plot_normal = lambda mu,sigma,ratio : plt.plot(X_axis, ratio*norm.pdf(X_axis,mu,sigma))
sns.kdeplot(X)
plot_normal(mu1_hat, sigma1_hat,pi1_hat)
plot_normal(mu2_hat, sigma2_hat,pi2_hat)
plt.show()


## E-step
#gamma1 = pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
#gamma2 = pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)
#total = gamma1 + gamma2
#gamma1 /= total
#gamma2 /= total
#
## M-step: Update parameters
#mu1_hat = np.sum(gamma1 * X) / np.sum(gamma1)
#mu2_hat = np.sum(gamma2 * X) / np.sum(gamma2)
#sigma1_hat = np.sqrt(np.sum(gamma1 * (X - mu1_hat)**2) / np.sum(gamma1))
#sigma2_hat = np.sqrt(np.sum(gamma2 * (X - mu2_hat)**2) / np.sum(gamma2))
#pi1_hat = np.mean(gamma1)
#pi2_hat = np.mean(gamma2)

num_epochs = 1000
log_likelihoods = []

plt.ion()
for epoch in range(num_epochs):
    print(f'epoch {epoch}...       ',end='\r')
    # E-step: Compute responsibilities
    gamma1 = pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
    gamma2 = pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)
    total = gamma1 + gamma2
    gamma1 /= total
    gamma2 /= total
     
    # M-step: Update parameters
    mu1_hat = np.sum(gamma1 * X) / np.sum(gamma1)
    mu2_hat = np.sum(gamma2 * X) / np.sum(gamma2)
    sigma1_hat = np.sqrt(np.sum(gamma1 * (X - mu1_hat)**2) / np.sum(gamma1))
    sigma2_hat = np.sqrt(np.sum(gamma2 * (X - mu2_hat)**2) / np.sum(gamma2))
    pi1_hat = np.mean(gamma1)
    pi2_hat = np.mean(gamma2)
     
    #print(f'N1~({mu1_hat:.2f},{sigma1_hat:.2f}), N2~({mu2_hat:.2f},{sigma2_hat:.2f})',end='              \r')

    # Compute log-likelihood
    log_likelihood = np.sum(np.log(pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
                                   + pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)))

    log_likelihoods.append(log_likelihood)

    if epoch < 20 or epoch % 20 == 19:
        sns.kdeplot(X, label='Data KDE')
        plot_normal(mu1_hat, sigma1_hat, pi1_hat)
        plot_normal(mu2_hat, sigma2_hat, pi2_hat)
        plt.legend()
        plt.draw()
        plt.pause(0.25)
        plt.clf()
plt.ioff()

print(f'N1~({mu1_hat:.4f},{sigma1_hat:.4f}), N2~({mu2_hat:.4f},{sigma2_hat:.4f})')

# Plot log-likelihood values over epochs
plt.plot(range(1, num_epochs+1), log_likelihoods)
plt.xlabel('Epoch')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs. Epoch')
#plt.show()



# Plot the final estimated density
X_sorted = np.sort(X)
density_estimation =   pi1_hat * norm.pdf(X_sorted, mu1_hat, sigma1_hat) \
                     + pi2_hat * norm.pdf(X_sorted, mu2_hat, sigma2_hat)
  
#plt.plot(X_sorted, gaussian_kde(X_sorted)(X_sorted), color='green', linewidth=2)
plt.figure()
sns.kdeplot(X_sorted)
plot_normal(mu1_hat, sigma1_hat, pi1_hat)
plot_normal(mu2_hat, sigma2_hat, pi2_hat)
plt.plot(X_axis, 0.25*norm.pdf(X_axis,mu1,sigma1), color="C1", linestyle='--')
plt.plot(X_axis, 0.75*norm.pdf(X_axis,mu2,sigma2), color="C2", linestyle='--')
plt.plot(X_sorted, density_estimation, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Density Estimation of X')
plt.legend(['Kernel Density Estimation','gmm1','gmm2','distribution 1','distribution 2','Mixture Density'])
plt.show()







