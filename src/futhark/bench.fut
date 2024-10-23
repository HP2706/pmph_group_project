import "lib/github.com/diku-dk/sorts/radix_sort"


let sort_i32 = radix_sort i32.num_bits i32.get_bit

let main (arr: []i32) = sort_i32 arr
