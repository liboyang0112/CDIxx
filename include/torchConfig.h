/**
 * @file 	torchConfig.h
 * @brief A config file header.
 * @author Boyang Li
 */
/**
 * @brief 	CDIxxinXsys Library.
 */

#ifndef __torchCONFIG
#define __torchCONFIG

#define DECINT(x,y) int x = y;
#define DECBOOL(x,y) bool x = y;
#define DECREAL(x,y) double x = y;
#define DECSTR(x,y) const char* x = y;

#define INTVAR(F) \
F(noise_size, 100)F(batch_size,64)F(n_epochs,30)F(check_point_every, 200)F(n_sample_per_check_point,10)F(log_interval,10)
#define BOOLVAR(F) \
F(restore, 0)
#define REALVAR(F) \

#define STRVAR(F) \
F(data_folder, "./data")

class torchConfig
{
public:
  BOOLVAR(DECBOOL)
  INTVAR(DECINT)
  REALVAR(DECREAL)
  STRVAR(DECSTR)
  torchConfig(const char* configfile);
  void print();
  ~torchConfig(){};
};
#endif
