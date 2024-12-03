# Step 1: Calculate the decimal day for Day 14 at 23:47
outlier_day = 14
outlier_time = "23:47"  # 23 hours 47 minutes
outlier_decimal_day = outlier_day + time_to_decimal(outlier_time)

# Step 2: Calculate the fitted height for this outlier
outlier_fitted_height = sinusoid(outlier_decimal_day, amplitude, frequency, phase, offset)

# Step 3: Define the outlier residual (2 feet above the fitted value)
outlier_residual = outlier_fitted_height + 2 - outlier_fitted_height

# Step 4: Add the outlier residual to the histogram
residuals_with_outlier = np.append(residuals, outlier_residual)  # Append the outlier to the residuals array

# Step 5: Plot the histogram with the outlier
plt.hist(residuals_with_outlier, bins=15, range=(-3.5, 3.5), alpha=0.5, edgecolor='white', density=True)

# Plot the Gaussian curve again
std_line = np.linspace(-5*residual_std, 5*residual_std, 1000)
plt.plot(std_line, gaussian(std_line, residual_mean, residual_std), color='red')

# Step 6: Add the outlier point to the histogram
plt.scatter(outlier_residual, 0, color='green', zorder=5, label="Outlier (Day 14, 23:47, +2 feet)")

# Annotate the outlier point
plt.text(outlier_residual + 0.1, 0.02, f'Outlier ({outlier_residual:.2f})', color='green')

# Labels and title
plt.xlabel('Residuals (ft)')
plt.ylabel('Density')
plt.title('Residual Histogram and Gaussian with Outlier')

# Add a legend
plt.legend()

# Show the plot
plt.grid()
plt.show()



def event_probability(x, mu, s):
    # outlier_residual in this context is the value of the event
    # residual_mean in this context is the gaussian mean
    # residual_std in this context is the gaussian standard deviation
    z = np.fabs((x - mu)/s)
    def zfunc(z):
        return 0.5*(1.0 + erf(z/2**0.5))

    #Return the probability of getting
    #an event of magnitude >=x
    return 1.0 - (zfunc(z) - zfunc(-1*z))
    
outlier_probability = event_probability(outlier_residual, residual_mean, residual_std)
print(f'The probability of the Tsunami having a 2-feet outlier is {outlier_probability * 100}%')
