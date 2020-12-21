#ifndef _MagnetometetNetworkInt8_H_
#define _MagnetometetNetworkInt8_H_


#include <ModelInterface.h>

class MagnetometetNetworkInt8 : public ModelInterface<int8_t>
{
	public:
		 MagnetometetNetworkInt8();
		 void forward();
};


#endif
