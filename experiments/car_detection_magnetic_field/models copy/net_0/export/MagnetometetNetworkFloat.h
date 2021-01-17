#ifndef _MagnetometetNetworkFloat_H_
#define _MagnetometetNetworkFloat_H_


#include <ModelInterface.h>

class MagnetometetNetworkFloat : public ModelInterface<float>
{
	public:
		 MagnetometetNetworkFloat();
		 void forward();
};


#endif
