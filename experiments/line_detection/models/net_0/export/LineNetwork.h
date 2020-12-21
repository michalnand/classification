#ifndef _LineNetwork_H_
#define _LineNetwork_H_


#include <ModelInterface.h>

class LineNetwork : public ModelInterface<int8_t>
{
	public:
		 LineNetwork();
		 void forward();
};


#endif
