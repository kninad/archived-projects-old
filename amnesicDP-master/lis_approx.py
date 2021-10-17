import math
import bisect
import random
import timeit
from collections import defaultdict


# Time = O(n^2)
# Space = O(n)
def standard_lis(nums):
    n = len(nums)
    vals = [1 for i in range(n)]     # LIS array
    dp = [1 for i in range(n)]
    lis = 0

    for i in range(n):        
        for l in range(0, i):
            if nums[i] > nums[l] and vals[i] < vals[l] + 1:
                vals[i] = vals[l] + 1
        dp[i] = max(vals[:i+1])

    # print(vals)
    # print(dp)
    return max(vals)

# Time = O(n log n)
# Space = O(n)
def binary_lis(nums):

    lis = [nums[0]] # lis[i] = smallest element that starts an LIS of length i
    for i in range(1,len(nums)):
        binarySearchIndex = bisect.bisect(lis,nums[i])
        # Avoid duplicate LIS
        if binarySearchIndex > 0 and lis[binarySearchIndex-1] == nums[i]:
            continue
        if binarySearchIndex == len(lis):
            lis.append(nums[i])
        else:
            lis[binarySearchIndex] = nums[i]
    
    return len(lis)


# gives incorrect result
def ss_lis(nums):
    n = len(nums)
    dp = [1 for i in range(n)]

    for i in range(1,n):
        for l in range(0,i):
            if nums[l] < nums[i]:
                score = dp[l] + 1
            else:
                score = dp[l]
            dp[i] = max(dp[i], score)
    
    print(dp)
    return dp[n-1]


# abandon ship
def approx_lis(nums, eps, k):
    if (k>=n):
        return binary_lis(nums)

    n = len(nums)
    # k = math.floor(1/eps)
    app_lis = 0     # approximation
    vals = [1 for i in range(n)]
    dp = [1 for i in range(n)]     # LIS array

    # for i<=k compute in the standard way
    # this can be made efficient
    for i in range(k):
        dp[i] = standard_lis(nums[:i+1])

    for i in range(k+1, n):
        curr_idx = i-1
        mult_factor = 1
        sampling_list = []
        counter = 0    
        
        while(curr_idx >= 0):   
            if counter < k:
                sampling_list.append(curr_idx)    
                curr_idx -= mult_factor
                counter += 1
            else:
                counter = 0
                mult_factor += 1

        for l in sampling_list:
        # for l in range(0, i):
            if nums[i] > nums[l] and vals[i] < vals[l] + 1:
                vals[i] = vals[l] + 1
        dp[i] = max(vals[:i+1])

    return app_lis


def amdp_basic(nums):
    n = len(nums)
    s_arr = [i for i in range(n+1)]
    w_arr = [i+1 for i in range(n)]
    for t in range(1, n):
        minVal = s_arr[t]
        for i in range(t):
            # print(i,t)
            if nums[i] < nums[t]:
                minVal = min(minVal, s_arr[i] + w_arr[t-1] - w_arr[i])
        s_arr[t] = minVal

    # Special case: last element == n index
    for i in range(n):
        s_arr[n] = min(s_arr[n], s_arr[i] + w_arr[n-1] - w_arr[i])

    return n - s_arr[-1]


# Discard probability from Sec 1.2
def prob(i, t, n):
    threshold = math.log10(n)
    if(t-i < threshold):
        return 0
    else:
        return 1.0/(t-i)


def amdp_approx(nums):
    n = len(nums)
    r_active = {}
    r_active[0] = 0
    # r_active[1] = 0
    maxSize = 0

    start_time = timeit.default_timer()
    for t in range(1, n):
        minVal = t

        for i in r_active.keys():
            # print(i,t)
            if nums[i] < nums[t]:
                minVal = min(minVal, r_active[i] + t - (i+1))
        r_active[t] = minVal
        
        # perform discard step
        norm_factor = sum([prob(i,t,n) for i in r_active.keys()])
        r_set = list(r_active.keys())
        for i in r_set:
            tmp = random.random()
            if tmp < prob(i, t, n)/norm_factor :
                r_active.pop(i, None)
                # could return to a temp var and assert that its not None
                # for double checking that key='i' exists in dict
        
        maxSize = max(maxSize, len(r_active.keys()))
        
        

    # Special case: last element == n index
    minVal_n = n
    for i in r_active.keys():        
        minVal_n = min(minVal_n, r_active[i] + n - (i+1))    
    r_active[n] = minVal_n

    stop_time = timeit.default_timer()
    
    ans = n - r_active[n]
    exec_time = stop_time - start_time
    
    return ans, maxSize, exec_time


