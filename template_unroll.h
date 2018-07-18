#ifndef SSJ_TEMPLATE_UNROLL_H
#define SSJ_TEMPLATE_UNROLL_H

/* Copyright 2014-2015 Willi Mann
 *
 * This file is part of set_sim_join.
 *
 * set_sim_join is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Foobar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with set_sim_join.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/mpl/vector/vector10.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/for_each.hpp>


namespace algo_template_unroll_d2 {
	using boost::mpl::vector;
	using boost::mpl::begin;
	using boost::mpl::next;
	using boost::mpl::end;
	using boost::mpl::if_;
	using boost::mpl::deref;
	using boost::mpl::distance;
	using boost::is_same;

	template <typename Execute>
	struct end_tag {
		static Algorithm * get_algo(
				std::vector<bool> & d1Pattern,
				std::vector<bool> & d2Pattern,
				typename Execute::algoParams algo_params) {
			assert(false);
			return NULL;
		}
	};

	template <class Classes1, class Classes2, class Execute>
		struct combine {




			template <class Class1It, class Class2It>
				class Generate;

			template <class Class1It, class Class2It>
				struct Next {
					typedef typename if_< 
						              is_same< 
						                typename end<Classes1>::type,
								        typename next<Class1It>::type
									  >,
									  typename if_<
										  is_same<
										      typename end<Classes2>::type,
									          typename next<Class2It>::type
									      >,
								          end_tag<Execute>,
								          Generate<
									          typename begin<Classes1>::type, 
								              typename next<Class2It>::type
										  >
									  >::type,
									  Generate<
			                             typename next<Class1It>::type,
									     Class2It
									  >
									>::type type;
				};

			template <class Class1It = typename begin<Classes1>::type,
					 class Class2It = typename begin<Classes2>::type >
						 struct Generate {

							 typedef typename Next<Class1It, Class2It>::type nexttype;
							 static Algorithm * get_algo( 
									 std::vector<bool> & d1Pattern,
									 std::vector<bool> & d2Pattern,
									 typename Execute::algoParams algo_params) {

								 typedef typename distance<
									     typename begin<Classes1>::type,
								         Class1It
									 >::type c1distance;

								 typedef typename distance<
									      typename begin<Classes2>::type,
										  Class2It
									 >::type c2distance;

								 Algorithm * algo = Execute::template get_algo<
									        typename deref<Class1It>::type,
									        typename deref<Class2It>::type
										 >(d1Pattern, d2Pattern, 
												 c1distance::value, c2distance::value, algo_params);
								 if(algo != NULL) {
									 return algo;
								 } else {
									 return nexttype::get_algo(d1Pattern, d2Pattern, algo_params);
								 }
							 }
						 };

		};
}

namespace algo_template_unroll_d3 {
	using boost::mpl::vector;
	using boost::mpl::begin;
	using boost::mpl::next;
	using boost::mpl::end;
	using boost::mpl::if_;
	using boost::mpl::deref;
	using boost::mpl::distance;
	using boost::is_same;

	template <typename Execute>
	struct end_tag {
		static Algorithm * get_algo(
				std::vector<bool> & d1Pattern,
				std::vector<bool> & d2Pattern,
				std::vector<bool> & d3Pattern,
				typename Execute::algoParams algo_params) {
			assert(false);
			return NULL;
		}
	};

	template <class Classes1, class Classes2, class Classes3, class Execute>
		struct combine {


			template <class Class1It, class Class2It, class Class3It>
				class Generate;

			template <class Class1It, class Class2It, class Class3It>
				struct Next {
					typedef typename if_< 
						              is_same< 
						                typename end<Classes1>::type,
								        typename next<Class1It>::type
									  >,
									  typename if_<
										  is_same<
										      typename end<Classes2>::type,
									          typename next<Class2It>::type
									      >,
								          typename if_<
										      is_same<
										          typename end<Classes3>::type,
									              typename next<Class3It>::type
									          >,
								              end_tag<Execute>,
								              Generate<
									              typename begin<Classes1>::type,
											      typename begin<Classes2>::type,
								                  typename next<Class3It>::type
										      >
									      >::type,
								          Generate<
									          typename begin<Classes1>::type,
								              typename next<Class2It>::type,
											  Class3It
										  >
									  >::type,
									  Generate<
			                             typename next<Class1It>::type,
									     Class2It,
										 Class3It
									  >
									>::type type;
				};

			template <class Class1It = typename begin<Classes1>::type,
					 class Class2It = typename begin<Classes2>::type,
					 class Class3It = typename begin<Classes3>::type>
						 struct Generate {

							 typedef typename Next<Class1It, Class2It, Class3It>::type nexttype;
							 static Algorithm * get_algo( 
									 std::vector<bool> & d1Pattern,
									 std::vector<bool> & d2Pattern,
									 std::vector<bool> & d3Pattern,
									 typename Execute::algoParams algo_params
									 ) {

								 typedef typename distance<
									     typename begin<Classes1>::type,
								         Class1It
									 >::type c1distance;

								 typedef typename distance<
									      typename begin<Classes2>::type,
										  Class2It
									 >::type c2distance;

								 typedef typename distance<
									      typename begin<Classes3>::type,
										  Class3It
									 >::type c3distance;

								 Algorithm * algo = Execute::template get_algo<
									        typename deref<Class1It>::type,
									        typename deref<Class2It>::type,
									        typename deref<Class3It>::type
										 >(d1Pattern, d2Pattern, d3Pattern, 
												 c1distance::value, c2distance::value, c3distance::value, algo_params);
								 if(algo != NULL) {
									 return algo;
								 } else {
									 return nexttype::get_algo(d1Pattern, d2Pattern, d3Pattern, algo_params);
								 }
							 }
						 };

		};
}

namespace algo_template_unroll_d4 {
	using boost::mpl::vector;
	using boost::mpl::begin;
	using boost::mpl::next;
	using boost::mpl::end;
	using boost::mpl::if_;
	using boost::mpl::deref;
	using boost::mpl::distance;
	using boost::is_same;

	template <typename Execute>
	struct end_tag {
		static Algorithm * get_algo(
				std::vector<bool> & d1Pattern,
				std::vector<bool> & d2Pattern,
				std::vector<bool> & d3Pattern,
				std::vector<bool> & d4Pattern,
				typename Execute::algoParams algo_params) {
			assert(false);
			return NULL;
		}
	};

	template <class Classes1, class Classes2, class Classes3, class Classes4, class Execute>
		struct combine {


			template <class Class1It, class Class2It, class Class3It, class Class4It>
				class Generate;

			template <class Class1It, class Class2It, class Class3It, class Class4It>
				struct Next {
					typedef typename if_< 
						              is_same< 
						                typename end<Classes1>::type,
								        typename next<Class1It>::type
									  >,
									  typename if_<
										  is_same<
										      typename end<Classes2>::type,
									          typename next<Class2It>::type
									      >,
								          typename if_<
										      is_same<
										          typename end<Classes3>::type,
									              typename next<Class3It>::type
									          >,
								              typename if_<
												  is_same<
													  typename end<Classes4>::type,
													  typename next<Class4It>::type
												  >,
												  end_tag<Execute>,
												  Generate<
													  typename begin<Classes1>::type,
													  typename begin<Classes2>::type,
													  typename begin<Classes3>::type,
													  typename next<Class4It>::type
												  >
											  >::type,
								              Generate<
									              typename begin<Classes1>::type,
											      typename begin<Classes2>::type,
								                  typename next<Class3It>::type,
												  Class4It
										      >
									      >::type,
								          Generate<
									          typename begin<Classes1>::type,
								              typename next<Class2It>::type,
											  Class3It,
											  Class4It
										  >
									  >::type,
									  Generate<
			                             typename next<Class1It>::type,
									     Class2It,
										 Class3It,
										 Class4It
									  >
									>::type type;
				};

			template <class Class1It = typename begin<Classes1>::type,
					 class Class2It = typename begin<Classes2>::type,
					 class Class3It = typename begin<Classes3>::type,
					 class Class4It = typename begin<Classes4>::type>
						 struct Generate {

							 typedef typename Next<Class1It, Class2It, Class3It, Class4It>::type nexttype;
							 static Algorithm * get_algo( 
									 std::vector<bool> & d1Pattern,
									 std::vector<bool> & d2Pattern,
									 std::vector<bool> & d3Pattern,
									 std::vector<bool> & d4Pattern,
									 typename Execute::algoParams algo_params
									 ) {

								 typedef typename distance<
									     typename begin<Classes1>::type,
								         Class1It
									 >::type c1distance;

								 typedef typename distance<
									      typename begin<Classes2>::type,
										  Class2It
									 >::type c2distance;

								 typedef typename distance<
									      typename begin<Classes3>::type,
										  Class3It
									 >::type c3distance;

								 typedef typename distance<
									      typename begin<Classes4>::type,
										  Class4It
									 >::type c4distance;

								 Algorithm * algo = Execute::template get_algo<
									        typename deref<Class1It>::type,
									        typename deref<Class2It>::type,
									        typename deref<Class3It>::type,
									        typename deref<Class4It>::type
										 >(d1Pattern, d2Pattern, d3Pattern, d4Pattern,
												 c1distance::value, c2distance::value, c3distance::value, c4distance::value,
												 algo_params);
								 if(algo != NULL) {
									 return algo;
								 } else {
									 return nexttype::get_algo(d1Pattern, d2Pattern, d3Pattern, d4Pattern, algo_params);
								 }
							 }
						 };

		};
}

namespace algo_template_unroll_d5 {
	using boost::mpl::vector;
	using boost::mpl::begin;
	using boost::mpl::next;
	using boost::mpl::end;
	using boost::mpl::if_;
	using boost::mpl::deref;
	using boost::mpl::distance;
	using boost::is_same;

	template<typename Execute>
	struct end_tag {
		static Algorithm * get_algo(
				std::vector<bool> & d1Pattern,
				std::vector<bool> & d2Pattern,
				std::vector<bool> & d3Pattern,
				std::vector<bool> & d4Pattern,
				std::vector<bool> & d5Pattern,
				typename Execute::algoParams algo_params) {
			assert(false);
			return NULL;
		}
	};

	template <class Classes1, class Classes2, class Classes3, class Classes4, class Classes5, class Execute>
		struct combine {


			template <class Class1It, class Class2It, class Class3It, class Class4It, class Class5It>
				class Generate;

			template <class Class1It, class Class2It, class Class3It, class Class4It, class Class5It>
				struct Next {
					typedef typename if_< 
						              is_same< 
						                typename end<Classes1>::type,
								        typename next<Class1It>::type
									  >,
									  typename if_<
										  is_same<
										      typename end<Classes2>::type,
									          typename next<Class2It>::type
									      >,
								          typename if_<
										      is_same<
										          typename end<Classes3>::type,
									              typename next<Class3It>::type
									          >,
								              typename if_<
												  is_same<
													  typename end<Classes4>::type,
													  typename next<Class4It>::type
												  >,
												  typename if_<
													  is_same<
														  typename end<Classes5>::type,
														  typename next<Class5It>::type
													  >,
													  end_tag<Execute>,
													  Generate<
														  typename begin<Classes1>::type,
														  typename begin<Classes2>::type,
														  typename begin<Classes3>::type,
														  typename begin<Classes4>::type,
														  typename next<Class5It>::type
													  >
												  >::type,
												  Generate<
													  typename begin<Classes1>::type,
													  typename begin<Classes2>::type,
													  typename begin<Classes3>::type,
													  typename next<Class4It>::type,
													  Class5It
												  >
											  >::type,
								              Generate<
									              typename begin<Classes1>::type,
											      typename begin<Classes2>::type,
								                  typename next<Class3It>::type,
												  Class4It,
												  Class5It
										      >
									      >::type,
								          Generate<
									          typename begin<Classes1>::type,
								              typename next<Class2It>::type,
											  Class3It,
											  Class4It,
											  Class5It
										  >
									  >::type,
									  Generate<
			                             typename next<Class1It>::type,
									     Class2It,
										 Class3It,
										 Class4It,
										 Class5It
									  >
									>::type type;
				};

			template <class Class1It = typename begin<Classes1>::type,
					 class Class2It = typename begin<Classes2>::type,
					 class Class3It = typename begin<Classes3>::type,
					 class Class4It = typename begin<Classes4>::type,
					 class Class5It = typename begin<Classes5>::type>
						 struct Generate {

							 typedef typename Next<Class1It, Class2It, Class3It, Class4It, Class5It>::type nexttype;
							 static Algorithm * get_algo( 
									 std::vector<bool> & d1Pattern,
									 std::vector<bool> & d2Pattern,
									 std::vector<bool> & d3Pattern,
									 std::vector<bool> & d4Pattern,
									 std::vector<bool> & d5Pattern,
									 typename Execute::algoParams algo_params
									 ) {

								 typedef typename distance<
									     typename begin<Classes1>::type,
								         Class1It
									 >::type c1distance;

								 typedef typename distance<
									      typename begin<Classes2>::type,
										  Class2It
									 >::type c2distance;

								 typedef typename distance<
									      typename begin<Classes3>::type,
										  Class3It
									 >::type c3distance;

								 typedef typename distance<
									      typename begin<Classes4>::type,
										  Class4It
									 >::type c4distance;

								 typedef typename distance<
									      typename begin<Classes5>::type,
										  Class5It
									 >::type c5distance;

								 Algorithm * algo = Execute::template get_algo<
									        typename deref<Class1It>::type,
									        typename deref<Class2It>::type,
									        typename deref<Class3It>::type,
									        typename deref<Class4It>::type,
									        typename deref<Class5It>::type
										 >(d1Pattern, d2Pattern, d3Pattern, d4Pattern, d5Pattern,
												 c1distance::value, c2distance::value, c3distance::value, c4distance::value, c5distance::value, 
												 algo_params);
								 if(algo != NULL) {
									 return algo;
								 } else {
									 return nexttype::get_algo(d1Pattern, d2Pattern, d3Pattern, d4Pattern, d5Pattern, algo_params);
								 }
							 }
						 };

		};
}
#endif
