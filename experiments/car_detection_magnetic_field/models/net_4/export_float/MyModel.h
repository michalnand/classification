#ifndef _MyModel_H_
#define _MyModel_H_


#include <ModelInterface.h>

class MyModel : public ModelInterface<float>
{
	public:
		 MyModel();
		 void forward();
};


#endif
