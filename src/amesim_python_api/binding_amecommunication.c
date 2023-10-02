#include <Python.h>
#include <stdio.h>
#include <string.h>

#include "amecommunication_socket.h"
#include "amecommunication_shm.h"

double *copyPyListToBuffer(PyObject *list)
{
   int i, size;
   double *output;
   PyObject *tmp_obj;

   if(!PyList_Check(list))
      return NULL;
   size = PyList_Size(list);
   if(size == 0)
      return NULL;
   
   output = (double *)calloc((size_t)size, sizeof(double));
   for(i = 0; i < size; i++)
   {
      tmp_obj = PyList_GetItem(list, i);
      if(PyFloat_Check(tmp_obj))
      {
         output[i] = PyFloat_AsDouble(tmp_obj);
      }
      else if(PyLong_Check(tmp_obj))
      {
         output[i] = (double)PyLong_AsLong(tmp_obj);
      }
      else
      {
         free(output);
         return NULL;
      }
   }
   return output;
}

PyObject *copyBufferToPyList(double *vec, int size)
{
   int i;
   PyObject *list, *tmp_obj;
   list = PyList_New(0);
   if(!PyList_Check(list))
      return NULL;
   for(i = 0; i < size; i++)
   {
      tmp_obj = PyFloat_FromDouble(vec[i]);
      PyList_Append(list, tmp_obj);
      Py_DECREF(tmp_obj);
   }
   return list;
}

static PyObject * amecommunication_sockclose(PyObject *self, PyObject *args)
{
   PyObject *h;
   amesock_conn_attributes *connection;
   int ret;
   
   if (!PyArg_ParseTuple(args, "O", &h))
      return NULL;
   connection = (amesock_conn_attributes*)PyCapsule_GetPointer(h, NULL);

   if(connection) {
      ret = amesock_close(connection);
      free(connection);
      return Py_BuildValue("i", ret);
   }
   else {
      return NULL;
   }
}

static PyObject * amecommunication_sockinit(PyObject *self, PyObject *args)
{
   char *server_name;
   int isserver, port, input_size, output_size, ret;
   amesock_conn_attributes * connection;

   connection = NULL;
   if (!PyArg_ParseTuple(args, "isiii", &isserver, &server_name, &port, &input_size, &output_size))
      return NULL;

   connection = (amesock_conn_attributes *)malloc(sizeof(amesock_conn_attributes));
   
   if(!connection) {
      PyErr_Format(PyExc_MemoryError, "unable to allocate %zu bytes", sizeof(amesock_conn_attributes));
      return NULL;
   }
   
   ret = amesock_init(connection, isserver, server_name, port, input_size, output_size);

   return Py_BuildValue("iO", ret, PyCapsule_New(connection, NULL, amecommunication_sockclose));
}

static PyObject * amecommunication_sockexchange(PyObject *self, PyObject *args)
{
   PyObject *h;
   amesock_conn_attributes *connection;
   int ret;
   PyObject *input, *output, *return_val;
   double *input_vec, *output_vec;

   if (!PyArg_ParseTuple(args, "OO", &h, &input))
      return NULL;
   
   connection = (amesock_conn_attributes*)PyCapsule_GetPointer(h, NULL);

   input_vec = copyPyListToBuffer(input);
   if(input_vec == NULL)
   {
      return NULL;
   }
   output_vec = (double *)calloc((size_t)connection->outputSize, sizeof(double));
   ret = amesock_exchange(connection, input_vec, output_vec);
   
   output = copyBufferToPyList(output_vec, connection->outputSize);
   return_val = Py_BuildValue("iO", ret, output);
   free(input_vec);
   Py_DECREF(output);
   return return_val;
}

/********************************************************
 *      COMMUNICATION THROW SHARED MEMORY               *
 ********************************************************/
 static PyObject * amecommunication_shmclose(PyObject *self, PyObject *args)
{
   PyObject *h;
   ameshm_conn_attributes *connection;
   int ret;
   
   if (!PyArg_ParseTuple(args, "O", &h))
      return NULL;
   connection = (ameshm_conn_attributes*)PyCapsule_GetPointer(h, NULL);

   if(connection) {
      ret = ameshm_close(connection);
      free(connection);
      return Py_BuildValue("i", ret);
   }
   else {
      return NULL;
   }
}
 
 static PyObject * amecommunication_shminit(PyObject *self, PyObject *args)
{
   char *name;
   int ismaster, input_size, output_size, ret;
   ameshm_conn_attributes * connection;

   connection = NULL;
   if (!PyArg_ParseTuple(args, "isii", &ismaster, &name, &input_size, &output_size))
      return NULL;

   connection = (ameshm_conn_attributes *)malloc(sizeof(ameshm_conn_attributes));

   if(!connection) {
      PyErr_Format(PyExc_MemoryError, "unable to allocate %zu bytes", sizeof(ameshm_conn_attributes));
      return NULL;
   }

   ret = ameshm_init(connection, ismaster, name, input_size, output_size);

   return Py_BuildValue("iO", ret, PyCapsule_New(connection, NULL, amecommunication_shmclose));
}

static PyObject * amecommunication_shmexchange(PyObject *self, PyObject *args)
{
   PyObject *h;
   ameshm_conn_attributes *connection;
   int ret;
   PyObject *input, *output, *return_val;
   double *input_vec, *output_vec;

   if (!PyArg_ParseTuple(args, "OO", &h, &input))
      return NULL;
   
   connection = (ameshm_conn_attributes*)PyCapsule_GetPointer(h, NULL);

   input_vec = copyPyListToBuffer(input);
   if(input_vec == NULL)
   {
      return NULL;
   }

   output_vec = (double *)calloc((size_t)connection->outputSize, sizeof(double));
   ret = ameshm_exchange(connection, input_vec, output_vec);
   
   output = copyBufferToPyList(output_vec, connection->outputSize);
   return_val = Py_BuildValue("iO", ret, output);
   free(input_vec);
   Py_DECREF(output);
   return return_val;
}

static PyMethodDef Methods[] = {
   {"sockinit",  amecommunication_sockinit, METH_VARARGS,
   "Initialize socket connection."},
   {"sockexchange",  amecommunication_sockexchange, METH_VARARGS,
   "Exchange datas throw socket connection."},
   {"sockclose",  amecommunication_sockclose, METH_VARARGS,
   "Close socket connection."},
   {"shminit",  amecommunication_shminit, METH_VARARGS,
   "Initialize sharedmemory connection."},
   {"shmexchange",  amecommunication_shmexchange, METH_VARARGS,
   "Exchange datas throw sharedmemory connection."},
   {"shmclose",  amecommunication_shmclose, METH_VARARGS,
   "Close sharedmemory connection."},
   {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "binding_amecommunication",   /* name of module */
    "Gateway module to the model DLL user co-simulation API functions", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    Methods
};

PyMODINIT_FUNC PyInit_binding_amecommunication(void)
{
   (void)PyModule_Create(&Module);
}

