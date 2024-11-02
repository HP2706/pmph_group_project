import "lib/github.com/diku-dk/sorts/radix_sort"


let sort_u64 = radix_sort u64.num_bits u64.get_bit

-- this is so ugly 
let main (arr: []u64) : []u64 = 
  sort_u64 arr