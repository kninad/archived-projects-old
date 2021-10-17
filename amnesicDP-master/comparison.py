from lis_approx import standard_lis, binary_lis, amdp_basic, amdp_approx

def compare(nums):
    ans1 = standard_lis(nums)
    ans2 = binary_lis(nums)
    ans3 = amdp_basic(nums)
    ans4 = amdp_approx(nums)
    return ans1, ans2, ans3, ans4





def main():
    fileName = "./test_file.txt"
    with open(fileName, 'r') as rfile:
        arrays = rfile.readlines()
        num_lists = []
        for arr in arrays:
            num_lists.append([int(x) for x in arr.split(',')])

    print (num_lists)

    for tmp in num_lists:
        print(compare(tmp))

if __name__ == "__main__":
    main()